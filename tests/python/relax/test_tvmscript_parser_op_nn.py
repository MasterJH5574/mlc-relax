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

from typing import Optional, Union

import tvm
import tvm.script
import tvm.testing
from tvm import IRModule, relax
from tvm.script import relax as R


def _check(
    parsed: Union[relax.Function, IRModule],
    expect: Optional[Union[relax.Function, IRModule]],
):
    test = parsed.script(show_meta=True)
    roundtrip_mod = tvm.script.from_source(test)
    tvm.ir.assert_structural_equal(parsed, roundtrip_mod)
    if expect:
        tvm.ir.assert_structural_equal(parsed, expect)


def test_conv2d():
    @R.function
    def foo(
        x: R.Tensor((2, 3, 228, 228), "float32"), w: R.Tensor((16, 3, 5, 5), "float32")
    ) -> R.Tensor((2, 16, 224, 224), "float16"):
        gv: R.Tensor((2, 16, 224, 224), "float16") = R.nn.conv2d(x, w, out_dtype="float16")
        return gv

    x = relax.Var("x", R.Tensor([2, 3, 228, 228], "float32"))
    w = relax.Var("w", R.Tensor([16, 3, 5, 5], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x, w]):
        gv = bb.emit(relax.op.nn.conv2d(x, w, out_dtype="float16"))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_max_pool2d():
    @R.function
    def foo(
        x: R.Tensor((1, 1, 32, 32), dtype="float32")
    ) -> R.Tensor((1, 1, 30, 30), dtype="float32"):
        gv: R.Tensor((1, 1, 30, 30), dtype="float32") = R.nn.max_pool2d(x, pool_size=(3,))
        return gv

    x = relax.Var("x", R.Tensor([1, 1, 32, 32], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.nn.max_pool2d(x, pool_size=(3,)))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_adaptive_avg_pool2d():
    @R.function
    def foo(x: R.Tensor((2, 64, 8, 9), "float32")) -> R.Tensor((2, 64, 7, 7), "float32"):
        gv: R.Tensor((2, 64, 7, 7), "float32") = R.nn.adaptive_avg_pool2d(x, output_size=(7, 7))
        return gv

    x = relax.Var("x", R.Tensor((2, 64, 8, 9), dtype="float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.nn.adaptive_avg_pool2d(x, output_size=(7, 7)))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_gelu():
    @R.function
    def foo(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
        gv: R.Tensor((2, 3), "float32") = R.nn.gelu(x)
        return gv

    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.nn.gelu(x))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_softmax():
    @R.function
    def foo(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
        gv: R.Tensor((2, 3), "float32") = R.nn.softmax(x)
        return gv

    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.nn.softmax(x))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_batch_norm():
    @R.function
    def foo(
        x: R.Tensor((2, 4, 3, 3), dtype="float32"),
        gamma: R.Tensor((4,), dtype="float32"),
        beta: R.Tensor((4,), dtype="float32"),
        moving_mean: R.Tensor((4,), dtype="float32"),
        moving_var: R.Tensor((4,), dtype="float32"),
    ) -> R.Tuple(
        R.Tensor((2, 4, 3, 3), dtype="float32"),
        R.Tensor((4,), dtype="float32"),
        R.Tensor((4,), dtype="float32"),
    ):
        gv: R.Tuple(
            R.Tensor((2, 4, 3, 3), dtype="float32"),
            R.Tensor((4,), dtype="float32"),
            R.Tensor((4,), dtype="float32"),
        ) = R.nn.batch_norm(x, gamma, beta, moving_mean, moving_var, axis=1)
        return gv

    x = relax.Var("x", R.Tensor((2, 4, 3, 3), "float32"))
    gamma = relax.Var("gamma", R.Tensor((4,), "float32"))
    beta = relax.Var("beta", R.Tensor((4,), "float32"))
    moving_mean = relax.Var("moving_mean", R.Tensor((4,), "float32"))
    moving_var = relax.Var("moving_var", R.Tensor((4,), "float32"))

    bb = relax.BlockBuilder()
    with bb.function("foo", [x, gamma, beta, moving_mean, moving_var]):
        gv = bb.emit(relax.op.nn.batch_norm(x, gamma, beta, moving_mean, moving_var, axis=1))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_layer_norm():
    @R.function
    def foo(
        x: R.Tensor((2, 3, 4, 5), "float32"),
        gamma: R.Tensor((4, 5), "float32"),
        beta: R.Tensor((4, 5), "float32"),
    ) -> R.Tensor((2, 3, 4, 5), "float32"):
        gv: R.Tensor((2, 3, 4, 5), "float32") = R.nn.layer_norm(x, gamma, beta, axes=[-2, -1])
        return gv

    x = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    gamma = relax.Var("gamma", R.Tensor((4, 5), "float32"))
    beta = relax.Var("beta", R.Tensor((4, 5), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x, gamma, beta]):
        gv = bb.emit(relax.op.nn.layer_norm(x, gamma, beta, axes=[-2, -1]))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_dropout():
    @R.function
    def foo(
        x: R.Tensor((2, 3), "float32")
    ) -> R.Tuple(R.Tensor((2, 3), "float32"), R.Tensor((2, 3), "float32")):
        gv: R.Tuple(R.Tensor((2, 3), "float32"), R.Tensor((2, 3), "float32")) = R.nn.dropout(
            x, rate=0.5
        )
        return gv

    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", [x]):
        gv = bb.emit(relax.op.nn.dropout(x))
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


if __name__ == "__main__":
    tvm.testing.main()
