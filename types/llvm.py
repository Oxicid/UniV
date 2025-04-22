# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import inspect
import typing
import unittest
import functools
import numpy as np
import ctypes as ct
from contextlib import contextmanager
from collections import namedtuple
from collections.abc import Callable

try:
    from llvmlite import ir, binding
    from llvmlite.ir import Constant
except ImportError:
    ir = None
    binding = None
    Constant = None

version = (0, 0, 3)

class ll:
    module: 'ir.Module' = None
    target: 'binding.Target' = None
    engine: 'binding.executionengine.ExecutionEngine' = None

    @staticmethod
    def compile():
        binding.initialize()
        binding.initialize_native_target()
        binding.initialize_native_asmprinter()

        ll.target = binding.Target.from_default_triple()
        target_machine = ll.target.create_target_machine()
        backing_mod = binding.parse_assembly(str(ll.module))
        backing_mod.verify()

        # Create pass manager for optimize code
        # pass_manager_builder = binding.PassManagerBuilder()
        # pass_manager_builder.opt_level = 0  # optimization O flag
        # pass_manager = binding.ModulePassManager()
        # pass_manager_builder.populate(pass_manager)
        # pass_manager.run(backing_mod)
        #
        ll.engine = binding.create_mcjit_compiler(backing_mod, target_machine)
        ll.engine.finalize_object()
        ll.engine.run_static_constructors()

    @staticmethod
    def initialize_types():
        ll.module = ir.Module(name="univ")
        bool_.type = ir.IntType(1)
        i8.type = ir.IntType(8)
        i16.type = ir.IntType(16)
        i32.type = ir.IntType(32)
        i64.type = ir.IntType(64)

        sum_n()
        array_sum()

class UniV:
    type = None

    @staticmethod
    def expected_type_check(other, expect):
        if expect != other:
            raise TypeError(f"Expect type {expect}, given {other}")

    @staticmethod
    def expected_subtype_check(other, expect):
        if not issubclass(other, expect):
            raise TypeError(f"Unsupported type: {other}")

class int_super_base(UniV):
    type: 'ir.IntType' = None
    signed = True
    var: 'ir.Type | ir.instructions.ICMPInstr | ir.Constant' = None
    _const_to_mutable_int = True
    def get_reference(self):
        return self.var.get_reference()  # noqa

    def to_bool(self):
        zero = Constant(self.type, 0)
        return bool_(fn.builder.icmp_signed('!=', self.value, zero))

    def and_(self, other, name=''):
        bool_state = fn.builder.and_(self.value, other.value, name)
        return bool_(bool_state)

    def or_(self, other, name=''):
        bool_state = fn.builder.or_(self.to_bool().value, other.to_bool().value, name)
        return bool_(bool_state)

    def __lt__(self, other):
        return bool_(fn.builder.icmp_signed('<', self.var, other.var))

    def __le__(self, other):
        return bool_(fn.builder.icmp_signed('<=', self.var, other.var))

    def __gt__(self, other):
        return bool_(fn.builder.icmp_signed('>', self.var, other.var))

    def __ge__(self, other):
        return bool_(fn.builder.icmp_signed('>=', self.var, other.var))

    def __eq__(self, other):
        return bool_(fn.builder.icmp_signed('==', self.var, other.var))

    def __ne__(self, other):
        return bool_(fn.builder.icmp_signed('!=', self.var, other.var))

    @property
    def value(self):
        return fn.builder.load(self.var)  # noqa

# base mutable ints
class int_base(int_super_base):
    def __init__(self, value, name=''):  # noqa
        if type(value) in (ir.IntType, ir.Argument, ir.instructions.ICMPInstr, ir.PhiInstr):
            if type(value) == ir.instructions.ICMPInstr and value.type.width != 1:
                raise NotImplementedError("Need implemented to_int")
            self.var = value
            if name:  # empty str need when manual set name before
                self.var.name = name
        elif isinstance(value, int_base):  # univ ints
            assert self.signed == value.signed
            assert type(self) == type(value)
            self.var = fn.builder.alloca(self.type, name=name)
            fn.builder.store(value.value, self.var)
        elif type(value) == ir.Constant:  # ir.Constant
            assert self.type == value.type
            if self._const_to_mutable_int:
                self.var = fn.builder.alloca(self.type, name=name)
                fn.builder.store(value, self.var)
            else:
                self.var = value
        elif isinstance(value, int):  # py int
            self.var = fn.builder.alloca(self.type, name=name)
            fn.builder.store(ir.Constant(self.type, value), self.var)
        elif type(value) == ir.instructions.Instruction:  # Add, Sub, and other
            assert value.type == self.type
            self.var = fn.builder.alloca(self.type, name=name)
            fn.builder.store(ir.Constant(self.type, value), self.var)
        elif type(value) == ir.GEPInstr: # gep
            # WARNING: Stored pointer instead use alloc to avoid stack overflow in loops
            assert value.type == self.type.as_pointer()
            self.var = value
        else:
            raise TypeError(f"Incorrect type: {type(value)}")

    @staticmethod
    def get_ctypes_analog():
        raise

    @classmethod
    def const(cls, value: int):
        assert isinstance(value, int)
        assert int_super_base._const_to_mutable_int
        int_super_base._const_to_mutable_int = False
        try:
            ret = cls(ir.Constant(cls.type, value))
        finally:
            int_super_base._const_to_mutable_int = True
        return ret

    @classmethod
    def phi(cls, name):
        return cls(fn.builder.phi(cls.type, name=name))

    def add_incoming(self, value: 'int_base', block):
        if value.type != self.type:
            raise TypeError(f"Type mismatch: {value.type} vs {self.type}")
        assert isinstance(self.var, ir.PhiInstr)
        self.var.add_incoming(value, block)

    @property
    def value(self):
        if isinstance(self.var, (ir.Constant, ir.PhiInstr)):
            return self.var
        return fn.builder.load(self.var)

    @value.setter
    def value(self, other):
        if isinstance(self.var, (ir.Argument, ir.Constant)):
            if isinstance(self.var, ir.Argument):
                raise NotImplementedError("Function integer arguments cannot be modified directly.")
            raise TypeError(f'{type(self.var)} is immutable type')
        elif isinstance(self.var, ir.instructions.ICMPInstr):
            raise TypeError(f'Pointer dereferencing cannot be applied to a variable resulting from a boolean expression')

        if isinstance(other, int):
            raise
            # return

        if isinstance(other, int_super_base):  # univ types with value property
            if self.signed != other.signed:
                # TODO: Improve behavior for boolean types
                # TODO: Auto convert (i32 + u16) -> i32=(i32 + i32(u16))
                # TODO: Error when assign (i16 <- u16)
                # TODO: Error when convert (unsigned + signed (non boolean))
                raise
            fn.builder.store(other.value, self.var)
        elif isinstance(other, ir.IntType):
            assert self.type.width == other.width
            # TODO: More checks
            fn.builder.store(other, self.var)
        elif isinstance(other, ir.instructions.Instruction):
            fn.builder.store(other, self.var)
        elif type(other) == ir.GEPInstr:
            raise AttributeError("Cannot assign to immutable value created from gep")
        else:
            raise TypeError(f"Incorrect type: {type(other)}, expect integers")

    def __add__(self, other):
        if not isinstance(other, int_super_base):
            raise TypeError(f"Incorrect type: {type(other)}, expect signed integer")
        if self.type != other.type:
            raise TypeError(f"Incorrect type: {other.type}, expect {self.type}")
        # TODO: Add python int
        b1 = type(self) == bool_
        b2 = type(other) == bool_
        if b1 or b2:
            raise NotImplementedError('Not implement __add__ for boolean type')
        new = fn.builder.add(self.value, other.value)
        return type(self)(new)

    def __iadd__(self, other):
        if not issubclass(type(other), int_super_base):
            raise TypeError(f"Incorrect type: {type(other)}, expect signed integer")
        if self.type != other.type:
            raise TypeError(f"Incorrect type: {other.type}, expect {self.type}")

        b1 = type(self) == bool_
        b2 = type(other) == bool_
        if b1 or b2:
            if b1:
                raise ValueError("Cannot apply in-place operation to boolean values")
            raise NotImplementedError('Not implement __add__ for boolean type')
        self.value = fn.builder.add(self.value, other.value)
        return self

class bool_(int_base):
    type = None

    @staticmethod
    def get_ctypes_analog():
        return ct.c_bool

class i8(int_base):
    type = None

    @staticmethod
    def get_ctypes_analog():
        return ct.c_int8

class i16(int_base):
    type = None

    @staticmethod
    def get_ctypes_analog():
        return ct.c_int16

class i32(int_base):
    type = None

    @staticmethod
    def get_ctypes_analog():
        return ct.c_int32

class i64(int_base):
    type = None

    @staticmethod
    def get_ctypes_analog():
        return ct.c_int64


# class Pointer: type = None

class Array(typing.Generic[typing.T], UniV):
    def __init__(self, ptr: 'ir.PointerType', univ_element_type, name=''):
        if not issubclass(univ_element_type, int_base):
            # TODO: Implement btypes
            raise TypeError(f"Unsupported parameter type: {univ_element_type}")

        self.ptr: 'ir.PointerType' = ptr
        self.ptr.name = name
        self.type = ir.PointerType(univ_element_type.type)
        self.univ_element_type = univ_element_type

    @property
    def element_type(self):
        return self.univ_element_type.type

    def sizeof(self) -> int:
        return sizeof(self.element_type)

    @staticmethod
    def is_array_type(typ):
        if typ == Array:
            raise TypeError('Raw pointers are not supported')
        if not (hasattr(typ, '__args__') and typ.__args__[0] == Array):
            return False
        if typ.__metadata__ and issubclass(typ.__metadata__[0], UniV):
            return True
        return False

    # For work with classmethod need Python 3.12+ (PEP 695)
    # @classmethod
    @staticmethod
    def get_ctypes_analog(typ):
        # assert cls.is_array_type()
        if typ == Array:
            # return ct.POINTER
            raise TypeError('Raw pointers are not supported')

        univ_type = typ.__metadata__[0]
        if issubclass(univ_type, int_base):
            return ct.POINTER(univ_type.get_ctypes_analog())
        raise TypeError(f'Type {typ.__metadata__} not supported')  # TODO: Implement other types

    @staticmethod
    def get_llvm_analog(typ):
        # assert cls.is_array_type()
        if typ == Array:
            # ir.PointerType(i8.type)
            raise TypeError('Raw pointers are not supported')

        univ_type = typ.__metadata__[0]
        if issubclass(univ_type, int_base):
            return ir.PointerType(univ_type.type)
        raise TypeError(f'Type {typ.__metadata__} not supported')

    def __getitem__(self, index: i32 | i64):
        gep = fn.builder.gep(self.ptr, [index.value])
        arr_get_item_name = self.ptr.name + '.array.getitem' if self.ptr.name else 'array.getitem'  # noqa
        return self.univ_element_type(gep, name=arr_get_item_name)

    def __setitem__(self, index: int, value):
        gep = fn.builder.gep(self.ptr, [ir.Constant(ir.IntType(32), index)])
        return fn.builder.store(value, gep)

    @staticmethod
    def __class_getitem__(item):
        return typing.Annotated[Array, item]

    def __repr__(self):
        return f"<Array of {self.univ_element_type}>"

Loop = namedtuple("Loop", ["index", "do_break"])

def const(i): return ir.Constant(ir.IntType(32), i)

def sizeof(typ: 'ir.Type') -> int:
    if isinstance(typ, ir.IntType):
        return typ.width // 8
    elif isinstance(typ, ir.FloatType):
        return 4 if typ == ir.FloatType() else 8
    elif isinstance(typ, ir.PointerType):
        return 8
    else:
        raise NotImplementedError(f"sizeof not implemented for {typ}")

def increment_index(builder, index):  # TODO: Delete this func
    return builder.add(index.var, const(1), name="incr")  # TODO: Replace const

def terminate(builder, target_block):
    builder.branch(target_block)

def cbranch(cond: bool_, true_br, false_br):
    return fn.builder.cbranch(cond.var, true_br, false_br)

@contextmanager
def for_range(count: i32):
    start = count.type(0)
    stop = count
    builder = fn.builder

    bb_cond = fn.builder.append_basic_block("for_range.cond")
    bb_body = fn.builder.append_basic_block("for_range.body")
    bb_end = fn.builder.append_basic_block("for_range.end")

    def do_break():
        builder.branch(bb_end)

    bb_start = builder.basic_block
    builder.branch(bb_cond)

    with builder.goto_block(bb_cond):
        index = i32.phi("loop.index")
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
    def get_params_from_signature(pyfunc, llvm_func: 'ir.Function'):
        signature = inspect.signature(pyfunc)
        params = signature.parameters.values()

        if len(params) != len(llvm_func.args):
            raise ValueError(f"Argument count mismatch: {len(params)} (Python) vs {len(llvm_func.args)} (LLVM) in the {pyfunc.__name__!r} function")

        univ_args: list[UniV] = []
        for param, arg in zip(params, llvm_func.args):
            # UniV.expected_subtype_check(param.annotation, UniV)

            univ_type = param.annotation
            if Array.is_array_type(univ_type):
                univ_element_type = univ_type.__metadata__[0]
                univ_args.append(Array(arg, univ_element_type, param.name))
            elif issubclass(univ_type, int_base):
                univ_args.append(univ_type(arg, param.name))
            else:
                raise NotImplementedError(f"Unsupported parameter type: {param}: {param.annotation} in the {pyfunc.__name__!r} function")
        assert all(isinstance(typ, UniV) for typ in univ_args)
        assert len(params) == len(univ_args)
        return univ_args

    @staticmethod
    def get_llvm_types_from_univ_signature(pyfunc):
        signature = inspect.signature(pyfunc)
        return_univ_element_type = signature.return_annotation
        if return_univ_element_type == inspect._empty:  # noqa
            raise NotImplementedError(f"Unsupported return void type in the {pyfunc.__name__!r} function")

        UniV.expected_subtype_check(return_univ_element_type, int_base)
        ret_type = return_univ_element_type.type

        arg_types = []
        params = signature.parameters.values()
        for param in signature.parameters.values():
            univ_type = param.annotation
            if Array.is_array_type(univ_type):
                arg_types.append(Array.get_llvm_analog(univ_type))
            elif issubclass(univ_type, int_base):
                arg_types.append(univ_type.type)
            else:
                raise NotImplementedError(f"Unsupported parameter type: {param.annotation} in the {pyfunc.__name__!r} function")

        assert isinstance(ret_type, ir.Type)
        assert all(isinstance(typ, ir.Type) for typ in arg_types)
        assert len(params) == len(arg_types)
        return ret_type, arg_types

    @staticmethod
    def get_ctypes_types_from_univ_signature(pyfunc):
        # TODO: Check exist pyfunc.__name__
        # TODO: Implement a function that from list((univ_type, llvm_type, ctype), ...) - creates dictionaries
        signature = inspect.signature(pyfunc)
        arg_types = []
        params = signature.parameters.values()
        for param in params:
            univ_type = param.annotation
            # Order is important because Array returns a typing.Annotation object, not a type
            if Array.is_array_type(univ_type):  # noqa
                arg_types.append(Array.get_ctypes_analog(univ_type))  # noqa
            elif issubclass(univ_type, int_base):
                arg_types.append(univ_type.get_ctypes_analog())
            else:
                if univ_type == inspect._empty:  # noqa
                    raise TypeError(f" Missing type specification for parameter {param.name!r} in the {pyfunc.__name__!r} function")
                raise NotImplementedError(f"Unsupported parameter type: {param.name}: {univ_type} in the {pyfunc.__name__!r} function")

        ret_univ_elem_type = signature.return_annotation
        if issubclass(ret_univ_elem_type, int_base):
            ret_type = ret_univ_elem_type.get_ctypes_analog()
        elif Array.is_array_type(ret_univ_elem_type):
            ret_type = Array.get_ctypes_analog(ret_univ_elem_type)
        else:
            raise NotImplementedError(f"Unsupported return type: {ret_univ_elem_type} in the {pyfunc.__name__!r} function")

        assert issubclass(ret_type, (ct._SimpleCData, ct._Pointer))  # noqa
        assert all(issubclass(typ, (ct._SimpleCData, ct._Pointer)) for typ in arg_types)  # noqa
        assert len(params) == len(arg_types)
        return ret_type, arg_types

    @staticmethod
    def get_c_func(func: Callable | str):
        if isinstance(func, str):
            if func in globals():
                func = globals()[func]
            else:
                raise AttributeError(f"Function {func!r} not found")
        func_ptr = ll.engine.get_function_address(func.__name__)
        if not func_ptr:
            raise AttributeError(f"Function {func.__name__!r} not found")
        ret_type, param_types = fn.get_ctypes_types_from_univ_signature(func)
        return ct.CFUNCTYPE(ret_type, *param_types)(func_ptr)


@fn()
def sum_n(n: i32) -> i32:
    acc = i32(0, name="acc")
    with for_range(n) as loop:
        acc+=loop.index
    fn.builder.ret(acc.value)  # TODO: Implement, with return value checks and auto converts

    return acc  # for return type warning disable

@fn()
def array_sum(arr: Array[i32], n: i32) -> i32:
    acc = i32(0, name="acc")
    with for_range(n) as loop:
        acc+=arr[loop.index]
    fn.builder.ret(acc.value)
    return acc  # for return type warning disable

class TestLLVM(unittest.TestCase):
    def test_sum_n(self):
        sum_n_ = fn.get_c_func(sum_n)
        self.assertEqual(sum_n_(10), 45)

    def test_array_sum(self):
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        array_ctypes = arr.ctypes.data_as(ct.POINTER(ct.c_int32))
        array_sum_ = fn.get_c_func(array_sum)
        self.assertEqual(array_sum_(array_ctypes, len(arr)), 15)

    @staticmethod
    def start():
        ll.compile()
        suite = unittest.TestLoader().loadTestsFromTestCase(TestLLVM)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        result.wasSuccessful()