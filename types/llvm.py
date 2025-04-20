# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import inspect
import functools
import ctypes as ct
from contextlib import contextmanager
from collections import namedtuple

try:
    from llvmlite import ir, binding
    from llvmlite.ir import Constant
except ImportError:
    ir = None
    binding = None
    Constant = None


class ll:
    module: 'ir.Module' = None
    target: 'binding.Target' = None
    engine: 'binding.executionengine.ExecutionEngine' = None

    @staticmethod
    def compile_and_run():
        binding.initialize()
        binding.initialize_native_target()
        binding.initialize_native_asmprinter()

        ll.target = binding.Target.from_default_triple()
        target_machine = ll.target.create_target_machine()
        backing_mod = binding.parse_assembly(str(ll.module))
        backing_mod.verify()

        # # Create pass manager for optimize code
        # pass_manager_builder = binding.PassManagerBuilder()
        # pass_manager_builder.opt_level = 2  # optimization O flag
        # pass_manager = binding.ModulePassManager()
        # pass_manager_builder.populate(pass_manager)
        # pass_manager.run(backing_mod)

        ll.engine = binding.create_mcjit_compiler(backing_mod, target_machine)
        ll.engine.finalize_object()
        ll.engine.run_static_constructors()

        func_ptr = ll.engine.get_function_address("sum_n")
        ret_type, param_types = fn.get_ctypes_types_from_univ_signature(sum_n)
        c_func = ct.CFUNCTYPE(ret_type, *param_types)(func_ptr)
        return c_func


    @staticmethod
    def initialize_types():
        ll.module = ir.Module(name="univ")
        bool_.type = ir.IntType(1)
        i8.type = ir.IntType(8)
        i16.type = ir.IntType(16)
        i32.type = ir.IntType(32)
        i64.type = ir.IntType(64)

        bool_c.type = ir.IntType(1)
        i8c.type = ir.IntType(8)
        i16c.type = ir.IntType(16)
        i32c.type = ir.IntType(32)
        i64c.type = ir.IntType(64)

        i8phi.type = ir.IntType(8)
        i16phi.type = ir.IntType(16)
        i32phi.type = ir.IntType(32)
        i64phi.type = ir.IntType(64)

        bool_arg.type = ir.IntType(1)
        i8arg.type = ir.IntType(8)
        i16arg.type = ir.IntType(16)
        i32arg.type = ir.IntType(32)
        i64arg.type = ir.IntType(64)

        sum_n()

class int_super_base:
    type: 'ir.IntType' = None
    var: 'ir.Type | ir.instructions.ICMPInstr' = None
    def get_reference(self):
        return self.var.get_reference()  # noqa

    def to_bool(self):
        zero = Constant(self.type, 0)
        return bool_(fn.builder.icmp_signed('!=', self.value, zero))

    def and_(self, other, name=''):
        bool_state = fn.builder.and_(self.to_bool().value, other.to_bool().value, name)
        return bool_(bool_state)

    def or_(self, other, name=''):
        bool_state = fn.builder.or_(self.to_bool().value, other.to_bool().value, name)
        return bool_(bool_state)

    def __lt__(self, other):
        value = fn.builder.icmp_signed('<', self.var, other.var)
        return bool_(value)  # TODO: Fix from other

    def __le__(self, other):
        return bool_(fn.builder.icmp_signed('<=', self.value, other.value))

    def __gt__(self, other):
        # return bool_(fn.builder.icmp_signed('>', self.value, other.value))
        return bool_(fn.builder.icmp_signed('>', self.var, other.var))  # TODO: Fix from other

    def __ge__(self, other):
        return bool_(fn.builder.icmp_signed('>=', self.value, other.value))

    def __eq__(self, other):
        # return bool_(fn.builder.icmp_signed('==', self.value, other.value))
        return bool_(fn.builder.icmp_signed('==', self.var, other.var))

    def __ne__(self, other):
        return bool_(fn.builder.icmp_signed('!=', self.value, other.value))

    @property
    def value(self):
        return fn.builder.load(self.var)  # noqa

# base mutable ints
class int_base(int_super_base):
    def __init__(self, value, name=''):
        if isinstance(value, ir.IntType):
            self.var = value
        elif isinstance(value, int_base):
            # TODO: More checks for avoid overflow
            var_ptr = fn.builder.alloca(self.type, name)
            fn.builder.store(value.value, var_ptr)  # int pointer
            self.var = var_ptr
        elif isinstance(value, int):
            self.var = fn.builder.alloca(self.type, name=name)
            fn.builder.store(ir.Constant(self.type, value), self.var)
        elif isinstance(value, ir.instructions.ICMPInstr):
            if value.type.width != 1:
                raise NotImplementedError("Not implemented to_int")
            self.var = value
        elif isinstance(value, ir.instructions.Instruction):
            self.var = fn.builder.alloca(self.type, name=name)
            fn.builder.store(ir.Constant(self.type, value), self.var)
        else:
            raise TypeError(f"Incorrect type: {type(value)}")

    @property
    def value(self):
        return fn.builder.load(self.var)

    @value.setter
    def value(self, other):
        if isinstance(other, int_super_base):
            fn.builder.store(other.value, self.var)
        elif hasattr(other, "type") and isinstance(other.type, ir.IntType):
            # TODO: More checks
            fn.builder.store(other, self.var)
        elif isinstance(other, ir.instructions.Instruction):
            fn.builder.store(other, self.var)
        else:
            raise TypeError(f"Incorrect type: {type(other)}, expect signed integer")

    def __add__(self, other):
        if not isinstance(other, int_super_base):
            raise TypeError(f"Incorrect type: {type(other)}, expect signed integer")
        if self.type != other.type:
            raise TypeError(f"Incorrect type: {other.type}, expect {self.type}")
        # TODO: Add python int
        new = fn.builder.add(self.value, other.value)
        return type(self)(new)

    def __iadd__(self, other):
        if not issubclass(type(other), int_super_base):
            raise TypeError(f"Incorrect type: {type(other)}, expect signed integer")
        if self.type != other.type:
            raise TypeError(f"Incorrect type: {other.type}, expect {self.type}")
        self.value = fn.builder.add(self.value, other.value)
        return self

class bool_(int_base): type = None
class i8(int_base): type = None
class i16(int_base): type = None
class i32(int_base): type = None
class i64(int_base): type = None

# const rvalue integers
class const_base(int_super_base):
    type: 'ir.IntType' = None
    def __init__(self, value):  # noqa
        self.var = self.type(value)  # noqa

    @property
    def value(self):
        return self.var

class bool_c(const_base): type = None
class i8c(const_base): type = None
class i16c(const_base): type = None
class i32c(const_base): type = None
class i64c(const_base): type = None

# phi integers
class int_phi_base(int_super_base):
    type: 'ir.IntType' = None

    def __init__(self, name):
        self.var = fn.builder.phi(self.type, name=name)

    @property
    def value(self):
        return self.var

    @value.setter
    def value(self, other):
        fn.builder.store(other.value, self.var)

    def add_incoming(self, value: int_base, block):
        if value.type != self.type:
            raise TypeError(f"Type mismatch: {value.type} vs {self.type}")
        self.var.add_incoming(value, block)  # noqa

class i8phi(int_phi_base): type = None
class i16phi(int_phi_base): type = None
class i32phi(int_phi_base): type = None
class i64phi(int_phi_base): type = None

class int_arg_base(int_super_base):
    def __init__(self, arg: 'ir.values.Argument', name=''):
        assert type(arg) == ir.values.Argument
        self.var: 'ir.values.Argument' = arg
        self.var.name = name

    @property
    def value(self):
        return fn.builder.load(self.var)

    @value.setter
    def value(self, other):
        raise NotImplementedError("Function arguments cannot be modified directly.")

class bool_arg(int_arg_base): type = None
class i8arg(int_arg_base): type = None
class i16arg(int_arg_base): type = None
class i32arg(int_arg_base): type = None
class i64arg(int_arg_base): type = None

def init_triplet_of_types():
    univ_llvm_ctypes: tuple[tuple[type, type, type] | tuple[type, type], ...] = (
        (bool_, ct.c_bool),
        (i8, ct.c_int8),
        (i16, ct.c_int16),
        (i32, ct.c_int32),
        (i64, ct.c_int64),
    )

Loop = namedtuple("Loop", ["index", "do_break"])

def const(i): return ir.Constant(ir.IntType(32), i)

def increment_index(builder, index):
    return builder.add(index.var, const(1), name="incr")

def terminate(builder, target_block):
    builder.branch(target_block)

def cbranch(cond: bool_, true_br, false_br):
    return fn.builder.cbranch(cond.var, true_br, false_br)

@contextmanager
def for_range(count: i32):
    start = count.type(0)
    stop = count
    builder = fn.builder

    bb_cond = fn.builder.append_basic_block("for.cond")
    bb_body = fn.builder.append_basic_block("for.body")
    bb_end = fn.builder.append_basic_block("for.end")

    def do_break():
        builder.branch(bb_end)

    bb_start = builder.basic_block
    builder.branch(bb_cond)

    with builder.goto_block(bb_cond):
        assert isinstance(count, (i32, i32arg))
        index = i32phi("loop.index")
        cbranch(index < stop, bb_body, bb_end)

    with builder.goto_block(bb_body):
        yield Loop(index, do_break)
        bb_body = builder.basic_block
        incr = increment_index(builder, index)
        terminate(builder, bb_cond)

    index.add_incoming(start, bb_start)
    index.add_incoming(incr, bb_body)

    builder.position_at_end(bb_end)

class fn:
    func = None
    type = None
    builder: 'ir.IRBuilder' = None

    @classmethod
    def call(cls):
        pass

    def __call__(self, pyfunc):
        @functools.wraps(pyfunc)
        def make_llvm_func():
            r_type, args = self.get_llvm_types_from_univ_signature(pyfunc)
            fn.type = ir.FunctionType(r_type, args)
            fn.func = ir.Function(ll.module, fn.type, name=pyfunc.__name__)
            fn.builder = ir.IRBuilder(self.func.append_basic_block(name="entry"))
            pyfunc(*self.get_params_from_signature(pyfunc, fn.func))
        return make_llvm_func

    @staticmethod
    def get_llvm_types_from_univ_signature(pyfunc):
        signature = inspect.signature(pyfunc)
        if signature.return_annotation == inspect._empty:  # noqa
            raise NotImplementedError(f"Unsupported return type: void")
        else:
            return_annotation = signature.return_annotation
            if return_annotation == i32:
                ret_type = return_annotation.type
            else:
                raise NotImplementedError(f"Unsupported return annotation type: {return_annotation}")

        arg_types = []
        for param in signature.parameters.values():
            if issubclass(param.annotation, int_base):
                arg_types.append(param.annotation.type)
            else:
                raise NotImplementedError(f"Unsupported parameter type: {param.annotation}")
        return ret_type, arg_types

    @staticmethod
    def get_params_from_signature(pyfunc, llvm_func: 'ir.Function'):
        signature = inspect.signature(pyfunc)
        params = signature.parameters.values()

        if len(params) != len(llvm_func.args):
            raise ValueError(f"Argument count mismatch: {len(params)} (Python) vs {len(llvm_func.args)} (LLVM)")

        univ_args = []
        for param, arg in zip(params, llvm_func.args):
            if param.annotation != i32:
                raise NotImplementedError(f"Unsupported parameter type: {param.annotation}")
            univ_args.append(i32arg(arg, param.name))
        return univ_args

    @staticmethod
    def get_ctypes_types_from_univ_signature(pyfunc):
        # TODO: Implement a function that from list((univ_type, llvm_type, ctype), ...) - creates dictionaries
        signature = inspect.signature(pyfunc)
        arg_types = []
        for param in signature.parameters.values():
            if param.annotation == i32:
                arg_types.append(ct.c_int32)
            else:
                raise NotImplementedError(f"Unsupported parameter type: {param.annotation}")

        return_annotation = signature.return_annotation
        if return_annotation == i32:
            ret_type = ct.c_int32
        else:
            raise NotImplementedError(f"Unsupported return type: {return_annotation}")
        return ret_type, arg_types

@fn()
def sum_n(n: i32) -> i32:
    acc = i32(0, name="acc")

    with for_range(n) as loop:
        acc+=loop.index

    fn.builder.ret(acc.value)  # TODO: Implement, with return value checks and auto converts

    return acc  # return type warning disable

class TestLLVM:
    # TODO: Implement unittest
    @staticmethod
    def test_sum_n():
        sum_n_ = ll.compile_and_run()
        assert sum_n_(10) == 45