# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import tvm
from tvm.relax.transform import LegalizeOps
from tvm.script import relax as R, tir as T
import tvm.testing


##################### Creation #####################


def test_full():
    # fmt: off
    @tvm.script.ir_module
    class Full:
        @R.function
        def main(v: R.Tensor((), "int32")) -> R.Tensor((2, 3), "int32"):
            gv: R.Tensor((2, 3), "int32") = R.full((2, 3), v, dtype="int32")
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(v: R.Tensor((), "int32")) -> R.Tensor((2, 3), "int32"):
            gv = R.call_tir(full, (v,), R.Tensor((2, 3), dtype="int32"))
            return gv

        @T.prim_func
        def full(rxplaceholder: T.Buffer((), "int32"), T_full: T.Buffer((T.int64(2), T.int64(3)), "int32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_full"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[()])
                    T.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = rxplaceholder[()]
    # fmt: on

    mod = LegalizeOps()(Full)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_full_constant_scalar_fill_value():
    # fmt: off
    @tvm.script.ir_module
    class Full:
        @R.function
        def main() -> R.Tensor((2, 3), "int32"):
            gv: R.Tensor((2, 3), "int32") = R.full((2, 3), R.const(3.5, "float32"), dtype="int32")
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((2, 3), "int32"):
            gv = R.call_tir(full, R.tuple(), R.Tensor((2, 3), dtype="int32"))
            return gv

        @T.prim_func
        def full(T_full: T.Buffer((T.int64(2), T.int64(3)), "int32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_full"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads()
                    T.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = 3
    # fmt: on

    mod = LegalizeOps()(Full)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_full_different_dtype():
    # fmt: off
    @tvm.script.ir_module
    class Full:
        @R.function
        def main(v: R.Tensor((), "int32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.full((2, 3), v, dtype="float32")
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(v: R.Tensor((), "int32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(full, (v,), R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func
        def full(rxplaceholder: T.Buffer((), "int32"), T_full: T.Buffer((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_full"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[()])
                    T.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = T.Cast("float32", rxplaceholder[()])
    # fmt: on

    mod = LegalizeOps()(Full)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_full_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Full:
        @R.function
        def main(dumb_param: R.Tensor(("m", "n")), v: R.Tensor((), "int32")) -> R.Tensor(("m", "n"), "int32"):
            m = T.var("int64")
            n = T.var("int64")
            gv: R.Tensor((m, n), "int32") = R.full((m, n), v, dtype="int32")
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(dumb_param: R.Tensor(("m", "n")), v: R.Tensor((), "int32")) -> R.Tensor(("m", "n"), "int32"):
            m = T.var("int64")
            n = T.var("int64")
            gv = R.call_tir(full, (v,), R.Tensor((m, n), dtype="int32"))
            return gv

        @T.prim_func
        def full(rxplaceholder: T.Buffer((), "int32"), var_T_full: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.var("int64")
            n = T.var("int64")
            T_full = T.match_buffer(var_T_full, [m, n], dtype="int32")
            for i0, i1 in T.grid(m, n):
                with T.block("T_full"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[()])
                    T.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = rxplaceholder[()]
    # fmt: on

    mod = LegalizeOps()(Full)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_full_like():
    # fmt: off
    @tvm.script.ir_module
    class FullLike:
        @R.function
        def main(x: R.Tensor((2, 3), "int32"), v: R.Tensor((), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.full_like(x, v)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "int32"), v: R.Tensor((), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(full, (v,), R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func
        def full(rxplaceholder: T.Buffer((), "float32"), T_full: T.Buffer((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_full"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[()])
                    T.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = rxplaceholder[()]
    # fmt: on

    mod = LegalizeOps()(FullLike)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_full_like_constant_scalar_fill_value():
    # fmt: off
    @tvm.script.ir_module
    class FullLike:
        @R.function
        def main(x: R.Tensor((2, 3), "int32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.full_like(x, R.const(-5, "float32"))
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "int32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(full, R.tuple(), R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func
        def full(T_full: T.Buffer((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_full"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads()
                    T.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = T.float32(-5)
    # fmt: on

    mod = LegalizeOps()(FullLike)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_full_like_different_dtype():
    # fmt: off
    @tvm.script.ir_module
    class FullLike:
        @R.function
        def main(x: R.Tensor((2, 3), "int32"), v: R.Tensor((), "float32")) -> R.Tensor((2, 3), "float64"):
            gv: R.Tensor((2, 3), "float64") = R.full_like(x, v, dtype="float64")
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "int32"), v: R.Tensor((), "float32")) -> R.Tensor((2, 3), "float64"):
            gv = R.call_tir(full, (v,), R.Tensor((2, 3), dtype="float64"))
            return gv

        @T.prim_func
        def full(rxplaceholder: T.Buffer((), "float32"), T_full: T.Buffer((T.int64(2), T.int64(3)), "float64")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_full"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[()])
                    T.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = T.Cast("float64", rxplaceholder[()])
    # fmt: on

    mod = LegalizeOps()(FullLike)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_full_like_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class FullLike:
        @R.function
        def main(x: R.Tensor(("m", "n"), "int32"), v: R.Tensor((), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv: R.Tensor((m, n), "float32") = R.full_like(x, v)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n"), "int32"), v: R.Tensor((), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv = R.call_tir(full, (v,), R.Tensor((m, n), dtype="float32"))
            return gv

        @T.prim_func
        def full(rxplaceholder: T.Buffer((), "float32"), var_T_full: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.var("int64")
            n = T.var("int64")
            T_full = T.match_buffer(var_T_full, [m, n], dtype="float32")
            for i0, i1 in T.grid(m, n):
                with T.block("T_full"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[()])
                    T.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = rxplaceholder[()]
    # fmt: on

    mod = LegalizeOps()(FullLike)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_ones():
    # fmt: off
    @tvm.script.ir_module
    class Ones:
        @R.function
        def main() -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.ones((2, 3), "float32")
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(ones, R.tuple(), R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func
        def ones(T_full: T.Buffer((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_full"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads()
                    T.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = T.float32(1)
    # fmt: on

    mod = LegalizeOps()(Ones)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_ones_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Ones:
        @R.function
        def main(dumb_param: R.Tensor(("m", "n"))) -> R.Tensor(("m", "n"), "float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv: R.Tensor((m, n), "float32") = R.ones((m, n), "float32")
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(dumb_param: R.Tensor(("m", "n"))) -> R.Tensor(("m", "n"), "float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv = R.call_tir(ones, R.tuple(), R.Tensor((m, n), dtype="float32"))
            return gv

        @T.prim_func
        def ones(var_T_full: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.var("int64")
            n = T.var("int64")
            T_full = T.match_buffer(var_T_full, [m, n], dtype="float32")
            for i0, i1 in T.grid(m, n):
                with T.block("T_full"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads()
                    T.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = T.float32(1)
    # fmt: on

    mod = LegalizeOps()(Ones)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_ones_like():
    # fmt: off
    @tvm.script.ir_module
    class OnesLike:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "int32"):
            gv: R.Tensor((2, 3), "int32") = R.ones_like(x, "int32")
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "int32"):
            gv = R.call_tir(ones, R.tuple(), R.Tensor((2, 3), dtype="int32"))
            return gv

        @T.prim_func
        def ones(T_full: T.Buffer((T.int64(2), T.int64(3)), "int32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_full"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads()
                    T.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = 1
    # fmt: on

    mod = LegalizeOps()(OnesLike)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_ones_like_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class OnesLike:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv: R.Tensor((m, n), "float32") = R.ones_like(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv = R.call_tir(ones, R.tuple(), R.Tensor((m, n), dtype="float32"))
            return gv

        @T.prim_func
        def ones(var_T_full: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.var("int64")
            n = T.var("int64")
            T_full = T.match_buffer(var_T_full, [m, n], dtype="float32")
            for i0, i1 in T.grid(m, n):
                with T.block("T_full"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads()
                    T.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = T.float32(1)
    # fmt: on

    mod = LegalizeOps()(OnesLike)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_zeros():
    # fmt: off
    @tvm.script.ir_module
    class Zeros:
        @R.function
        def main() -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.zeros((2, 3), "float32")
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(zeros, R.tuple(), R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func
        def zeros(T_full: T.Buffer((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_full"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads()
                    T.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = T.float32(0)
    # fmt: on

    mod = LegalizeOps()(Zeros)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_zeros_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Zeros:
        @R.function
        def main(dumb_param: R.Tensor(("m", "n"))) -> R.Tensor(("m", "n"), "float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv: R.Tensor((m, n), "float32") = R.zeros((m, n), "float32")
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(dumb_param: R.Tensor(("m", "n"))) -> R.Tensor(("m", "n"), "float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv = R.call_tir(zeros, R.tuple(), R.Tensor((m, n), dtype="float32"))
            return gv

        @T.prim_func
        def zeros(var_T_full: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.var("int64")
            n = T.var("int64")
            T_full = T.match_buffer(var_T_full, [m, n], dtype="float32")
            for i0, i1 in T.grid(m, n):
                with T.block("T_full"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads()
                    T.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = T.float32(0)
    # fmt: on

    mod = LegalizeOps()(Zeros)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_zeros_like():
    # fmt: off
    @tvm.script.ir_module
    class ZerosLike:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "int32"):
            gv: R.Tensor((2, 3), "int32") = R.zeros_like(x, "int32")
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "int32"):
            gv = R.call_tir(zeros, R.tuple(), R.Tensor((2, 3), dtype="int32"))
            return gv

        @T.prim_func
        def zeros(T_full: T.Buffer((T.int64(2), T.int64(3)), "int32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_full"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads()
                    T.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = 0
    # fmt: on

    mod = LegalizeOps()(ZerosLike)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_zeros_like_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class ZerosLike:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv: R.Tensor((m, n), "float32") = R.zeros_like(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.var("int64")
            n = T.var("int64")
            gv = R.call_tir(zeros, R.tuple(), R.Tensor((m, n), dtype="float32"))
            return gv

        @T.prim_func
        def zeros(var_T_full: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.var("int64")
            n = T.var("int64")
            T_full = T.match_buffer(var_T_full, [m, n], dtype="float32")
            for i0, i1 in T.grid(m, n):
                with T.block("T_full"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads()
                    T.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = T.float32(0)
    # fmt: on

    mod = LegalizeOps()(ZerosLike)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_tril():
    # fmt: off
    @tvm.script.ir_module
    class Tril:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor((2, 3, 4), "float32"):
            gv: R.Tensor((2, 3, 4), "float32") = R.tril(x, k=1)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor((2, 3, 4), "float32"):
            gv = R.call_tir(tril, (x,), R.Tensor((2, 3, 4), dtype="float32"))
            return gv

        @T.prim_func
        def tril(rxplaceholder: T.Buffer((T.int64(2), T.int64(3), T.int64(4)), "float32"), trilu: T.Buffer((T.int64(2), T.int64(3), T.int64(4)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(3), T.int64(4)):
                with T.block("trilu"):
                    i0_1, i1_1, i2_1 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(rxplaceholder[i0_1, i1_1, i2_1])
                    T.writes(trilu[i0_1, i1_1, i2_1])
                    trilu[i0_1, i1_1, i2_1] = T.Select(i2_1 - T.int64(1) <= i1_1, rxplaceholder[i0_1, i1_1, i2_1], T.float32(0))
    # fmt: on

    mod = LegalizeOps()(Tril)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_tril_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Tril:
        @R.function
        def main(x: R.Tensor(("m", "n", "k"), "int8")) -> R.Tensor(("m", "n", "k"), "int8"):
            m = T.var("int64")
            n = T.var("int64")
            k = T.var("int64")
            gv: R.Tensor((m, n, k), "int8") = R.tril(x, k=-2)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n", "k"), "int8")) -> R.Tensor(("m", "n", "k"), "int8"):
            m = T.var("int64")
            n = T.var("int64")
            k = T.var("int64")
            gv = R.call_tir(tril, (x,), R.Tensor((m, n, k), dtype="int8"))
            return gv

        @T.prim_func
        def tril(var_rxplaceholder: T.handle, var_trilu: T.handle):
            T.func_attr({"tir.noalias": True})
            k = T.var("int64")
            m = T.var("int64")
            n = T.var("int64")
            rxplaceholder = T.match_buffer(var_rxplaceholder, [m, n, k], dtype="int8")
            trilu = T.match_buffer(var_trilu, [m, n, k], dtype="int8")
            for i0, i1, i2 in T.grid(m, n, k):
                with T.block("trilu"):
                    i0_1, i1_1, i2_1 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(rxplaceholder[i0_1, i1_1, i2_1])
                    T.writes(trilu[i0_1, i1_1, i2_1])
                    trilu[i0_1, i1_1, i2_1] = T.Select(i2_1 + T.int64(2) <= i1_1, rxplaceholder[i0_1, i1_1, i2_1], T.int8(0))
    # fmt: on

    mod = LegalizeOps()(Tril)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_triu():
    # fmt: off
    @tvm.script.ir_module
    class Triu:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor((2, 3, 4), "float32"):
            gv: R.Tensor((2, 3, 4), "float32") = R.triu(x, k=1)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor((2, 3, 4), "float32"):
            gv = R.call_tir(triu, (x,), R.Tensor((2, 3, 4), dtype="float32"))
            return gv

        @T.prim_func
        def triu(rxplaceholder: T.Buffer((T.int64(2), T.int64(3), T.int64(4)), "float32"), trilu: T.Buffer((T.int64(2), T.int64(3), T.int64(4)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(3), T.int64(4)):
                with T.block("trilu"):
                    i0_1, i1_1, i2_1 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(rxplaceholder[i0_1, i1_1, i2_1])
                    T.writes(trilu[i0_1, i1_1, i2_1])
                    trilu[i0_1, i1_1, i2_1] = T.Select(i1_1 <= i2_1 - T.int64(1), rxplaceholder[i0_1, i1_1, i2_1], T.float32(0))
    # fmt: on

    mod = LegalizeOps()(Triu)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_triu_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Triu:
        @R.function
        def main(x: R.Tensor(("m", "n", "k"), "int8")) -> R.Tensor(("m", "n", "k"), "int8"):
            m = T.var("int64")
            n = T.var("int64")
            k = T.var("int64")
            gv: R.Tensor((m, n, k), "int8") = R.triu(x, k=-2)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n", "k"), "int8")) -> R.Tensor(("m", "n", "k"), "int8"):
            m = T.var("int64")
            n = T.var("int64")
            k = T.var("int64")
            gv = R.call_tir(triu, (x,), R.Tensor((m, n, k), dtype="int8"))
            return gv

        @T.prim_func
        def triu(var_rxplaceholder: T.handle, var_trilu: T.handle):
            T.func_attr({"tir.noalias": True})
            k = T.var("int64")
            m = T.var("int64")
            n = T.var("int64")
            rxplaceholder = T.match_buffer(var_rxplaceholder, [m, n, k], dtype="int8")
            trilu = T.match_buffer(var_trilu, [m, n, k], dtype="int8")
            for i0, i1, i2 in T.grid(m, n, k):
                with T.block("trilu"):
                    i0_1, i1_1, i2_1 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(rxplaceholder[i0_1, i1_1, i2_1])
                    T.writes(trilu[i0_1, i1_1, i2_1])
                    trilu[i0_1, i1_1, i2_1] = T.Select(i1_1 <= i2_1 + T.int64(2), rxplaceholder[i0_1, i1_1, i2_1], T.int8(0))
    # fmt: on

    mod = LegalizeOps()(Triu)
    tvm.ir.assert_structural_equal(mod, Expected)


##################### Datatype #####################


def test_astype():
    # fmt: off
    @tvm.script.ir_module
    class Astype:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor((2, 3, 4), "int32"):
            gv: R.Tensor((2, 3, 4), "int32") = R.astype(x, "int32")
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor((2, 3, 4), "int32"):
            gv = R.call_tir(cast, (x,), R.Tensor((2, 3, 4), dtype="int32"))
            return gv

        @T.prim_func
        def cast(rxplaceholder: T.Buffer((T.int64(2), T.int64(3), T.int64(4)), "float32"), compute: T.Buffer((T.int64(2), T.int64(3), T.int64(4)), "int32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(3), T.int64(4)):
                with T.block("compute"):
                    i0_1, i1_1, i2_1 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(rxplaceholder[i0_1, i1_1, i2_1])
                    T.writes(compute[i0_1, i1_1, i2_1])
                    compute[i0_1, i1_1, i2_1] = T.Cast("int32", rxplaceholder[i0_1, i1_1, i2_1])
    # fmt: on

    mod = LegalizeOps()(Astype)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_astype_input_constant_scalar():
    # fmt: off
    @tvm.script.ir_module
    class Astype:
        @R.function
        def main() -> R.Tensor((), "int32"):
            gv: R.Tensor((), "int32") = R.astype(R.const(1.5, "float32"), "int32")
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((), "int32"):
            gv: R.Tensor((), "int32") = R.const(1, "int32")
            return gv
    # fmt: on

    mod = LegalizeOps()(Astype)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_astype_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Astype:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "int32"):
            m = T.var("int64")
            n = T.var("int64")
            gv: R.Tensor((m, n), "int32") = R.astype(x, "int32")
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "int32"):
            m = T.var("int64")
            n = T.var("int64")
            gv = R.call_tir(cast, (x,), R.Tensor((m, n), dtype="int32"))
            return gv

        @T.prim_func
        def cast(var_rxplaceholder: T.handle, var_compute: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.var("int64")
            n = T.var("int64")
            rxplaceholder = T.match_buffer(var_rxplaceholder, [m, n], dtype="float32")
            compute = T.match_buffer(var_compute, [m, n], dtype="int32")
            for i0, i1 in T.grid(m, n):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.Cast("int32", rxplaceholder[i0_1, i1_1])
    # fmt: on

    mod = LegalizeOps()(Astype)
    tvm.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    tvm.testing.main()
