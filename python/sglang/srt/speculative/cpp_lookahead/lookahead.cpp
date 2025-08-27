#include "lookahead.h"

#include <limits>
#include <vector>

namespace lookahead {

struct Node {
    std::unordered_map<int32_t, int32_t> next;
};

Lookahead::Result fillResult(int last_token, int draft_token_num, std::vector<Node>& tree, int root) {
    Lookahead::Result info;
    std::vector<int32_t> prevs;
    info.token.reserve(draft_token_num);
    prevs.reserve(draft_token_num);
    std::queue<std::tuple<int32_t, int32_t, int32_t>> queue;
    info.token.emplace_back(last_token);
    prevs.emplace_back(-1);

    for (auto [token, next] : tree[root].next) {
        queue.emplace(token, next, 0);
    }
    while (queue.size()) {
        auto [token, next, prev] = queue.front();
        queue.pop();
        info.token.emplace_back(token);
        prevs.emplace_back(prev);
        for (auto [t, n] : tree[next].next) {
            queue.emplace(t, n, info.token.size() - 1);
        }
    }

    // zero padding to length
    while(info.token.size() < draft_token_num) {
        info.token.emplace_back(0);
        prevs.emplace_back(0);
    }

    int n = info.token.size();
    info.mask.resize(n * n, 0);
    info.mask[0] = 1;
    for (int i = 0; i < n; ++i) {
        if (prevs[i] != -1) {
            memcpy(&info.mask[i * n], &info.mask[prevs[i] * n], prevs[i] + 1);
        }
        info.mask[i * n + i] = 1;
    }

    return info;
}

Lookahead::Lookahead(size_t capacity, const Param& param) {
    param_ = param;
    nodes_.resize(capacity);
    for (auto& node : nodes_) {
        node_pool_.emplace_back(&node);
    }
    free_node_count_ = node_pool_.size();
    root_ = getNode();

    CHECK(param_.branch_length > 1) << " param_.branch_length: " << param_.branch_length;
    CHECK(param_.min_match_window_size > 0) << " min_match_window_size: " << param_.min_match_window_size;
    CHECK(param_.min_match_window_size <= param_.max_match_window_size) << " min_match_window_size: " << param_.min_match_window_size << " max_match_window_size: " << param_.max_match_window_size;
    CHECK(param_.max_match_window_size < param_.branch_length) << " max_match_window_size: " << param_.max_match_window_size << " branch_length: " << param_.branch_length;
    CHECK(param_.min_bfs_breadth > 0) << " min_bfs_breadth: " << param_.min_bfs_breadth;
    CHECK(param_.min_bfs_breadth <= param_.max_bfs_breadth) << " min_bfs_breadth: " << param_.min_bfs_breadth << " max_bfs_breadth: " << param_.max_bfs_breadth;
    CHECK(param_.draft_token_num > 0) << " draft_token_num: " << param_.draft_token_num;
    for (auto config : param_.batch_return_token_num) {
        if (config != std::numeric_limits<decltype(config)>::max()) {
            CHECK(config <= param_.draft_token_num) << " batch_return_token_num: " << config << " draft_token_num: " << param_.draft_token_num;
        }
    }
    for (auto config : param_.batch_min_match_window_size) {
        if (config != std::numeric_limits<decltype(config)>::max()) {
            CHECK(config >= param_.min_match_window_size) << " config: " << config << " min_match_window_size: " << param_.min_match_window_size;
            CHECK(config <= param_.max_match_window_size) << " config: " << config << " max_match_window_size: " << param_.max_match_window_size;
        }
    }

    quit_flag_ = false;
    insert_worker_ = std::thread(&Lookahead::insert, this);
}

Lookahead::~Lookahead() { 
    quit_flag_ = true;
    insert_queue_.close();
    insert_worker_.join();
}

std::vector<std::pair<TrieNode*, int32_t>> Lookahead::match(const std::vector<int32_t>& tokens, size_t batch_size) const {
    auto draft_token_num = param_.get_return_token_num(batch_size);
    auto min_match_window_size = param_.get_min_match_window_size(batch_size);
    auto max_match_window_size = param_.max_match_window_size;
    std::vector<std::pair<TrieNode*, int32_t>> result;
    result.reserve(param_.max_match_window_size - param_.min_match_window_size);
    for (int32_t match_window_size = std::min(tokens.size(), param_.max_match_window_size);
         match_window_size >= param_.min_match_window_size;
         --match_window_size) {
        auto start = tokens.data() + tokens.size() - match_window_size;
        auto end = start + match_window_size;
        auto cursor = root_;
        while (start != end) {
            auto iter = cursor->child.find(*start);
            if (iter == cursor->child.end()) {
                cursor = nullptr;
                break;
            }
            ++start;
            cursor = iter->second;
        }
        if (cursor) {
            result.emplace_back(std::make_pair(cursor, match_window_size));
        }
    }
    return result;
}

void Lookahead::squeeze(size_t count) {
    CHECK(node_pool_.size() >= free_node_count_ + count);
    while (count--) {
        auto last = global_lru_.back();
        global_lru_.pop_back();

        CHECK(last->child.empty());

        last->parent->lru.erase(last->parent_lru_pos);
        last->parent->sorted_children.erase(last);
        last->parent->child.erase(last->token);

        node_pool_[free_node_count_++] = last;
    }
}

void Lookahead::synchronize() const {
    while (!insert_queue_.empty()) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

void Lookahead::insert() {
    while (!quit_flag_) {
        std::vector<int32_t> data;
        if (!insert_queue_.dequeue(data)) {
            continue;
        }
        const auto* token = data.data();
        size_t size = data.size();
        std::unique_lock<std::mutex> lock(mutex_);

        for (size_t i = 0; i + param_.min_match_window_size < size; ++i) {
            auto start = token + i;
            auto end = start + std::min(size - i, param_.branch_length);

            if (end - start > free_node_count_) {
                squeeze(end - start - free_node_count_);
            }

            TrieNode* cursor = root_;
            path_.clear();
            while (start != end) {
                auto token = *start;
                auto iter = cursor->child.find(token);
                if (iter == cursor->child.end()) {
                    iter = cursor->child.insert({token, getNode()}).first;
                    auto node = iter->second;

                    cursor->lru.emplace_front(node);
                    global_lru_.emplace_back(node);

                    node->token = token;
                    node->parent = cursor;
                    node->parent_lru_pos = cursor->lru.begin();
                    node->global_lru_pos = --global_lru_.end();
                    node->freq = 1;
                    cursor->sorted_children.insert(node);
                } else {
                    auto node = iter->second;
                    cursor->sorted_children.erase(node);
                    node->freq++;
                    cursor->sorted_children.insert(node);
                    cursor->lru.splice(cursor->lru.begin(), cursor->lru, node->parent_lru_pos);
                }
                cursor = iter->second;
                path_.emplace_back(cursor);
                ++start;
            }

            for (auto it = path_.rbegin(); it != path_.rend(); ++it) {
                TrieNode* node = *it;
                global_lru_.splice(global_lru_.begin(), global_lru_, node->global_lru_pos);
            }
        }
    }
}

void Lookahead::asyncInsert(std::vector<std::vector<int32_t>>&& tokens) {
    for (auto&& token : tokens) {
        insert_queue_.enqueue(std::move(token));
    }
}

Lookahead::Result Lookahead::matchBFS(const std::vector<int32_t>& tokens, size_t batch_size) const {
    std::vector<std::pair<TrieNode*, int32_t>> nodes = match(tokens, batch_size);

    double bfs_breadth_scale = double(param_.max_bfs_breadth - param_.min_bfs_breadth) /
                               (param_.max_match_window_size - param_.min_match_window_size + 1);

    auto draft_token_num = param_.get_return_token_num(batch_size);
    std::vector<Node> tree(draft_token_num + 1);
    int root = 0;
    int cursor = 1;

    for (auto [node, depth] : nodes) {
        std::queue<std::tuple<int32_t, double, const TrieNode*>> queue;  // parent, bfs_breadth, node
        queue.push({root, (param_.max_match_window_size - depth) * bfs_breadth_scale + param_.min_bfs_breadth, node});
        while (queue.size() && cursor <= draft_token_num) {
            auto front = queue.front();
            queue.pop();

            auto parent = std::get<0>(front);
            auto cur_breadth = std::get<1>(front);
            auto iter = std::get<2>(front)->lru.begin();

            auto breadth = std::max(1, int32_t(cur_breadth));
            for (int i = 0; i < breadth && iter != std::get<2>(front)->lru.end() && cursor <= draft_token_num;
                 ++i, ++iter) {
                auto token = (*iter)->token;
                auto pos = -1;
                if (auto tit = tree[parent].next.find(token); tit != tree[parent].next.end()) {
                    pos = tit->second;
                } else {
                    pos = tree[parent].next.insert(std::make_pair(token, cursor++)).first->second;
                }
                queue.emplace(pos, cur_breadth - bfs_breadth_scale, *iter);
            }
        }
    }

    return fillResult(tokens.back(), draft_token_num + 1, tree, root);
}

Lookahead::Result Lookahead::matchProb(const std::vector<int32_t>& tokens, size_t batch_size) const {
    std::vector<std::pair<TrieNode*, int32_t>> nodes = match(tokens, batch_size);
    auto draft_token_num = param_.get_return_token_num(batch_size);
    struct CompareByLastDouble {
    bool operator()(
        const std::tuple< double, const TrieNode*, double>& a,// parent_pos,  node, final_prob
        const std::tuple< double, const TrieNode*, double>& b) const
    {
        return std::get<2>(a) < std::get<2>(b);  // 比较freq
    }
    };
    std::vector<Node> tree(draft_token_num + 1);
 
    int root = 0;
    int cursor = 1;
    int top_k = param_.max_bfs_breadth; 
    float temperature = 0.6;     
    for(auto [node,depth] : nodes){
        std::priority_queue<std::tuple< double, const TrieNode*, double>,
        std::vector<std::tuple< double, const TrieNode*, double>>,
        CompareByLastDouble> heap;
        double sum_freq = 0.0;
        int count = 0;
        std::vector<std::pair<TrieNode*, int32_t>> topk_children;
        for (auto* child : node->sorted_children) {
            sum_freq += static_cast<double>(child->freq) / temperature;
            topk_children.emplace_back(child, child->freq);
            if (++count >= top_k) break;
        }
        if (sum_freq <= 0) sum_freq = 1.0;
        // count = 0;
        // normalize and emplace
        for (const auto& [child, freq] : topk_children) {
            double norm_freq = (static_cast<double>(freq)/temperature) / sum_freq;
            heap.emplace(root, child, norm_freq);
        }
        while (!heap.empty() && cursor <= draft_token_num)
        {
            auto [parent, trie_node, prob] = heap.top();// parent_pos, node, final_prob
            heap.pop();
            auto token = trie_node->token; 
            int pos = -1;
            auto tit = tree[parent].next.find(token);
            if (tit != tree[parent].next.end()) {
                pos = tit->second;
                
            } else {
                pos = cursor++;
                tree[parent].next[token] = pos;
                // if (cursor >= (int)tree.size()) {
                //     tree.emplace_back();
                // }
            }
 
            double sum_freq = 0.0;
            std::vector<std::pair<TrieNode*, int32_t>> topk_children;
            int count = 0;
            for (auto* child : trie_node->sorted_children) {
                sum_freq += static_cast<double>(child->freq)/ temperature;
                topk_children.emplace_back(child, child->freq);
                if (++count >= top_k) break;
            }
            if (sum_freq <= 0) sum_freq = 1.0;
            count = 0;
            // normalize and emplace
            for (const auto& [child, freq] : topk_children) {
                double norm_freq = (static_cast<double>(freq)/ temperature) / sum_freq * prob;
                heap.emplace(pos, child, norm_freq);
            }
        }
    }

    return fillResult(tokens.back(), draft_token_num + 1, tree, root);
}

Lookahead::Result Lookahead::batchMatch(const std::vector<std::vector<int32_t>>& tokens) const {
    std::unique_lock<std::mutex> lock(mutex_);
    Result merged_result;
    for (const auto& tks : tokens) {
        Result res = matchBFS(tks, tokens.size());
        merged_result.token.insert(merged_result.token.end(), res.token.begin(), res.token.end());
        merged_result.mask.insert(merged_result.mask.end(), res.mask.begin(), res.mask.end());
    }
    return merged_result;
}

/*Lookahead::Result Lookahead::matchBFS_sort(const std::vector<int32_t>& tokens, size_t batch_size) {
    std::unique_lock<std::mutex> lock(mutex_);
    std::vector<std::pair<TrieNode*, int32_t>> nodes = match(tokens, batch_size);

    double bfs_breadth_scale = double(param_.max_bfs_breadth - param_.min_bfs_breadth) /
                               (param_.max_match_window_size - param_.min_match_window_size + 1);
    auto draft_token_num = param_.get_return_token_num(batch_size);//decoding length
    std::vector<Node> tree(draft_token_num + 1);
    
    int root = 0;
    int cursor = 1;
    for (auto [node, depth] : nodes) {
        std::queue<std::tuple<int32_t, double, const TrieNode*>> queue;  // parent, bfs_breadth, node
        queue.push({root, (param_.max_match_window_size - depth) * bfs_breadth_scale + param_.min_bfs_breadth, node});
        while (queue.size() && cursor <= draft_token_num) {
            auto front = queue.front();
            queue.pop();

            auto parent = std::get<0>(front);
            auto cur_breadth = std::get<1>(front);
            const TrieNode* trie_node = std::get<2>(front);
            //auto iter = std::get<2>(front)->lru.begin();
            auto breadth = std::max(1, int32_t(cur_breadth));
            std::vector<TrieNode*> sorted_children(trie_node->lru.begin(), trie_node->lru.end());
            // 2. 排序：频率降序
            std::sort(sorted_children.begin(), sorted_children.end(),
                    [](TrieNode* a, TrieNode* b) {
                        return a->freq > b->freq;
                    });
            for (int i = 0; i < breadth && i < sorted_children.size() && cursor <= draft_token_num;
                 ++i) {
                TrieNode* child = sorted_children[i];
                auto token = child->token;
                
                // 在 parent 的 next vector 里查找有没有这个 token
                auto it = std::find_if(
                    tree[parent].next.begin(),
                    tree[parent].next.end(),
                    [&](const std::pair<int32_t, int32_t>& p) { return p.first == token; }
                );
                auto pos = -1;
                if (it != tree[parent].next.end()) {
                    pos = it->second;
                } else {
                    pos = cursor++;
                    tree[parent].next.emplace_back(token, pos); 
                }

                queue.emplace(pos, cur_breadth - bfs_breadth_scale, child);
            }
        }
    }
    
    return fillResult(tokens.back(), draft_token_num + 1, tree, root);
}*/

void Lookahead::fillLowerTrangularMatrix(Result& info) {
    int n = info.token.size();
    info.mask.clear();
    info.mask.resize(n * n);
    for (int i = 0; i < n; ++i) {
        std::fill_n(info.mask.begin() + i * n, i + 1, uint8_t(1));
    }
}

void Lookahead::Result::truncate(size_t n) {
    if (n < token.size()) {
        int full_n = token.size();
        for (int i = 1; i < n; ++i) {
            memcpy(&mask[i * n], &mask[i * full_n], sizeof(mask[0]) * n);
        }
        token.resize(n);
        mask.resize(n * n);
    }
}

}  // namespace lookahead
