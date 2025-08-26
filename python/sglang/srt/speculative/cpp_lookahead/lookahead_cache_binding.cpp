#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lookahead.h"


PYBIND11_MODULE(lookahead_cache_cpp, m) {
    using namespace lookahead;
    namespace py = pybind11;
    m.doc() = "";

    py::class_<Lookahead>(m, "Lookahead")
        .def(
            py::init<size_t, const Param&>(),
            py::arg("capacity"),
            py::arg("param"))
        .def("async_insert", &Lookahead::async_insert, "")
        .def("matchBFS", &Lookahead::matchBFS, "")
        .def("matchProb", &Lookahead::matchProb, "")
        .def("reset", &Lookahead::reset, "")
        .def("synchronize", &Lookahead::synchronize, "");

    py::class_<Param>(m, "Param")
        .def(py::init<>())
        .def_readwrite("enable", &Param::enable)
        .def_readwrite("enable_router_mode", &Param::enable_router_mode)
        .def_readwrite("min_bfs_breadth", &Param::min_bfs_breadth)
        .def_readwrite("max_bfs_breadth", &Param::max_bfs_breadth)
        .def_readwrite("min_match_window_size", &Param::min_match_window_size)
        .def_readwrite("max_match_window_size", &Param::max_match_window_size)
        .def_readwrite("branch_length", &Param::branch_length)
        .def_readwrite("return_token_limit", &Param::return_token_limit)
        .def_readwrite("capacity", &Param::capacity)
        .def_readwrite("batch_min_match_window_size", &Param::batch_min_match_window_size)
        .def_readwrite("batch_return_token_num", &Param::batch_return_token_num)
        .def("get_return_token_num", &Param::get_return_token_num, "")
        .def("get_min_match_window_size", &Param::get_min_match_window_size, "")
        .def("parse", &Param::parse, "")
        .def("resetBatchMinMatchWindowSize", &Param::resetBatchMinMatchWindowSize, "")
        .def("resetBatchReturnTokenNum", &Param::resetBatchReturnTokenNum, "")
        .def("detail", &Param::detail, "");

    py::class_<Lookahead::Result>(m, "Result")
        .def(py::init<>())
        .def_readwrite("token", &Lookahead::Result::token)
        .def_readwrite("mask", &Lookahead::Result::mask)
        .def_readwrite("prev", &Lookahead::Result::prev)
        .def_readwrite("position", &Lookahead::Result::position)
        .def("truncate", &Lookahead::Result::truncate);
}
