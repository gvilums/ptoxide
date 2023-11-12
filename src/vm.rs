use std::collections::HashMap;

use crate::ast::MulMode;
use crate::ast::SpecialReg;

#[derive(Debug, Clone, Copy)]
pub enum Symbol {
    Function(usize),
    Variable(usize),
}

#[derive(Clone, Copy, Debug)]
pub enum StateSpace {
    // includes ptx global and const
    Global,
    // includes ptx local and param
    Stack,
    // includes ptx shared
    Shared,
}

#[derive(Clone, Copy, Debug)]
pub struct RegPred {
    id: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Reg8 {
    id: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Reg16 {
    id: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Reg32 {
    id: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Reg64 {
    id: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Reg128 {
    id: usize,
}

use crate::ast::PredicateOp;
use crate::ast::Type;

#[derive(Clone, Copy, Debug)]
pub enum RegOperand {
    Pred(RegPred),
    B8(Reg8),
    B16(Reg16),
    B32(Reg32),
    B64(Reg64),
    B128(Reg128),
    Special(SpecialReg),
}

#[derive(Clone, Copy, Debug)]
pub enum Constant {
    B128(u128),
    B64(u64),
    B32(u32),
    B16(u16),
    B8(u8),
    U64(u64),
    U32(u32),
    U16(u16),
    U8(u8),
    S64(i64),
    S32(i32),
    S16(i16),
    S8(i8),
    F64(f64),
    F32(f32),
    F16x2(f32, f32),
    F16(f32),
    Pred(bool),
}

#[derive(Clone, Copy, Debug)]
pub enum AddrOperand {
    Absolute(usize),
    AbsoluteReg(usize, Reg64),
    StackRelative(isize),
    StackRelativeReg(isize, Reg64),
    RegisterRelative(Reg64, isize),
}

#[derive(Clone, Copy, Debug)]
pub enum Instruction {
    Load(StateSpace, RegOperand, AddrOperand),
    Store(StateSpace, RegOperand, AddrOperand),
    Convert {
        dst_type: Type,
        src_type: Type,
        dst: RegOperand,
        src: RegOperand,
    },
    Move(Type, RegOperand, RegOperand),
    Const(RegOperand, Constant),
    Add(Type, RegOperand, RegOperand, RegOperand),
    Mul(Type, MulMode, RegOperand, RegOperand, RegOperand),
    ShiftLeft(Type, RegOperand, RegOperand, RegOperand),
    SetPredicate(Type, PredicateOp, RegPred, RegOperand, RegOperand),
    Jump {
        offset: isize,
    },
    SkipIf(RegPred, bool),
    Return,
}

#[derive(Clone, Copy, Debug)]
pub struct IPtr(pub usize);

#[derive(Clone, Debug)]
struct FrameMeta {
    return_addr: IPtr,
    frame_size: usize,
}

#[derive(Debug)]
struct ThreadState {
    iptr: IPtr,
    regs: Vec<Registers>,
    stack_data: Vec<u8>,
    frame_meta: Vec<FrameMeta>,
    // function_args: Vec<isize>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RegDesc {
    pred_count: usize,
    b8_count: usize,
    b16_count: usize,
    b32_count: usize,
    b64_count: usize,
    b128_count: usize,
}

impl RegDesc {
    pub fn alloc_pred(&mut self) -> RegPred {
        let id = self.pred_count;
        self.pred_count += 1;
        RegPred { id }
    }
    pub fn alloc_b8(&mut self) -> Reg8 {
        let id = self.b8_count;
        self.b8_count += 1;
        Reg8 { id }
    }
    pub fn alloc_b16(&mut self) -> Reg16 {
        let id = self.b16_count;
        self.b16_count += 1;
        Reg16 { id }
    }
    pub fn alloc_b32(&mut self) -> Reg32 {
        let id = self.b32_count;
        self.b32_count += 1;
        Reg32 { id }
    }
    pub fn alloc_b64(&mut self) -> Reg64 {
        let id = self.b64_count;
        self.b64_count += 1;
        Reg64 { id }
    }
    pub fn alloc_b128(&mut self) -> Reg128 {
        let id = self.b128_count;
        self.b128_count += 1;
        Reg128 { id }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FuncFrameDesc {
    pub iptr: IPtr,
    pub frame_size: usize,
    pub arg_size: usize,
    pub regs: RegDesc,
}

#[derive(Debug)]
pub struct Context {
    global_mem: Vec<u8>,
    instructions: Vec<Instruction>,
    descriptors: Vec<FuncFrameDesc>,
    symbol_map: HashMap<String, Symbol>,
}

#[derive(Debug)]
struct Registers {
    pred: Vec<bool>,
    b8: Vec<[u8; 1]>,
    b16: Vec<[u8; 2]>,
    b32: Vec<[u8; 4]>,
    b64: Vec<[u8; 8]>,
    b128: Vec<[u8; 16]>,
}

impl Registers {
    fn new(desc: RegDesc) -> Registers {
        Registers {
            pred: vec![false; desc.pred_count],
            b8: vec![[0; 1]; desc.b8_count],
            b16: vec![[0; 2]; desc.b16_count],
            b32: vec![[0; 4]; desc.b32_count],
            b64: vec![[0; 8]; desc.b64_count],
            b128: vec![[0; 16]; desc.b128_count],
        }
    }
}

macro_rules! generate_reg_functions {
    ($($t:ident, $get:ident, $set:ident, $field:ident, $n:expr);*) => {
        $(
            fn $get(&self, reg: $t) -> [u8; $n] {
                self.regs.last().unwrap().$field[reg.id]
            }

            fn $set(&mut self, reg: $t, value: [u8; $n]) {
                self.regs.last_mut().unwrap().$field[reg.id] = value;
            }
        )*
    };
}

macro_rules! generate_reg_functions2 {
    ($($t:ident, $get:ident, $set:ident, $field:ident, $t2:ident);*) => {
        $(
            fn $get(&self, reg: $t) -> $t2 {
                $t2::from_ne_bytes(self.regs.last().unwrap().$field[reg.id])
            }

            fn $set(&mut self, reg: $t, value: $t2) {
                self.regs.last_mut().unwrap().$field[reg.id] = value.to_ne_bytes();
            }
        )*
    };
}

impl ThreadState {
    fn new() -> ThreadState {
        ThreadState {
            iptr: IPtr(0),
            regs: Vec::new(),
            stack_data: Vec::new(),
            frame_meta: Vec::new(),
            // function_args: Vec::new(),
        }
    }

    fn get_pred(&self, reg: RegPred) -> bool {
        self.regs.last().unwrap().pred[reg.id]
    }

    fn set_pred(&mut self, reg: RegPred, value: bool) {
        self.regs.last_mut().unwrap().pred[reg.id] = value;
    }

    generate_reg_functions!(
        Reg8, get_b8, set_b8, b8, 1;
        Reg16, get_b16, set_b16, b16, 2;
        Reg32, get_b32, set_b32, b32, 4;
        Reg64, get_b64, set_b64, b64, 8;
        Reg128, get_b128, set_b128, b128, 16
    );

    generate_reg_functions2!(
        Reg8, get_u8, set_u8, b8, u8;
        Reg16, get_u16, set_u16, b16, u16;
        Reg32, get_u32, set_u32, b32, u32;
        Reg64, get_u64, set_u64, b64, u64;
        Reg128, get_u128, set_u128, b128, u128;

        Reg8, get_i8, set_i8, b8, i8;
        Reg16, get_i16, set_i16, b16, i16;
        Reg32, get_i32, set_i32, b32, i32;
        Reg64, get_i64, set_i64, b64, i64;
        Reg128, get_i128, set_i128, b128, i128;

        // Reg8, get_f8, set_f8, b8, f32;
        // Reg16, get_f16, set_f16, b16, f16;
        Reg32, get_f32, set_f32, b32, f32;
        Reg64, get_f64, set_f64, b64, f64
        // Reg128, get_f128, set_f128, b128, f64
    );

    fn iptr_fetch_incr(&mut self) -> IPtr {
        let ret = self.iptr;
        self.iptr.0 += 1;
        ret
    }

    fn frame_teardown(&mut self) {
        let frame_meta = self.frame_meta.pop().unwrap();
        self.stack_data
            .truncate(self.stack_data.len() - frame_meta.frame_size);
        self.regs.pop();
        self.iptr = frame_meta.return_addr;
    }

    fn frame_setup(&mut self, desc: FuncFrameDesc) {
        self.frame_meta.push(FrameMeta {
            return_addr: self.iptr,
            frame_size: desc.frame_size,
        });
        self.stack_data
            .resize(self.stack_data.len() + desc.frame_size, 0);
        self.regs.push(Registers::new(desc.regs));
        self.iptr = desc.iptr;
    }

    fn num_frames(&self) -> usize {
        self.frame_meta.len()
    }

    fn resolve_address(&self, addr: AddrOperand) -> usize {
        match addr {
            AddrOperand::Absolute(addr) => addr,
            AddrOperand::AbsoluteReg(addr, reg) => {
                let offset = u64::from_ne_bytes(self.get_b64(reg)) as isize;
                (addr as isize + offset) as usize
            }
            AddrOperand::StackRelative(offset) => {
                let frame_size = self.frame_meta.last().unwrap().frame_size as isize;
                (self.stack_data.len() as isize - frame_size + offset) as usize
            }
            AddrOperand::StackRelativeReg(offset, reg) => {
                let frame_size = self.frame_meta.last().unwrap().frame_size as isize;
                let reg_offset = u64::from_ne_bytes(self.get_b64(reg)) as isize;
                (self.stack_data.len() as isize - frame_size + offset + reg_offset) as usize
            }
            AddrOperand::RegisterRelative(reg, offset) => {
                let reg_base = u64::from_ne_bytes(self.get_b64(reg)) as isize;
                (reg_base + offset) as usize
            }
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum VmError {
    #[error("Invalid register operand {:?} for instruction {:?}", .1, .0)]
    InvalidOperand(Instruction, RegOperand),
    #[error("Parameter data did not match descriptor")]
    ParamDataSizeMismatch,
    #[error("Slice size mismatch")]
    SliceSizeMismatch(#[from] std::array::TryFromSliceError),
    #[error("Parse error: {0:?}")]
    ParseError(#[from] crate::ast::ParseErr),
    #[error("Compile error: {0:?}")]
    CompileError(#[from] crate::compiler::CompilationError),
}

#[derive(Clone, Copy, Debug)]
pub struct DevicePointer(u64);

pub enum Argument<'a> {
    Ptr(DevicePointer),
    U64(u64),
    U32(u32),
    Bytes(&'a [u8]),
}

#[derive(Clone, Copy, Debug)]
pub struct LaunchParams {
    func_id: usize,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
}

impl LaunchParams {
    pub fn new() -> LaunchParams {
        LaunchParams {
            func_id: 0,
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
        }
    }

    pub fn func(mut self, id: usize) -> LaunchParams {
        self.func_id = id;
        self
    }

    pub fn grid1d(mut self, x: u32) -> LaunchParams {
        self.grid_dim = (x, 1, 1);
        self
    }

    pub fn block1d(mut self, x: u32) -> LaunchParams {
        self.block_dim = (x, 1, 1);
        self
    }
}

impl Context {
    fn fetch_instr(&self, iptr: IPtr) -> Instruction {
        self.instructions[iptr.0]
    }

    #[cfg(test)]
    fn new_raw(program: Vec<Instruction>, descriptors: Vec<FuncFrameDesc>) -> Context {
        Context {
            global_mem: Vec::new(),
            instructions: program,
            descriptors,
            symbol_map: HashMap::new(),
        }
    }

    pub fn new() -> Context {
        Context {
            global_mem: Vec::new(),
            instructions: Vec::new(),
            descriptors: Vec::new(),
            symbol_map: HashMap::new(),
        }
    }

    pub fn new_with_module(module: &str) -> Result<Self, VmError> {
        let module = crate::ast::parse_program(module)?;
        let compiled = crate::compiler::compile(module)?;
        Ok(Self {
            global_mem: Vec::new(),
            instructions: compiled.instructions,
            descriptors: compiled.func_descriptors,
            symbol_map: compiled.symbol_map,
        })
    }

    pub fn load(&mut self, module: &str) -> Result<(), crate::ast::ParseErr> {
        let module = crate::ast::parse_program(module)?;
        let _compiled = crate::compiler::compile(module).unwrap();
        todo!()
    }

    pub fn alloc(&mut self, size: usize, align: usize) -> DevicePointer {
        // Calculate the next aligned position
        let aligned_ptr = (self.global_mem.len() + align - 1) & !(align - 1);
        // Resize the vector to ensure the space is allocated
        self.global_mem.resize(aligned_ptr + size, 0);
        // Return the device pointer to the aligned address
        DevicePointer(aligned_ptr as u64)
    }

    pub fn write(&mut self, ptr: DevicePointer, offset: usize, data: &[u8]) {
        let begin = ptr.0 as usize + offset;
        let end = begin + data.len();
        self.global_mem[begin..end].copy_from_slice(data);
    }

    pub fn read(&mut self, ptr: DevicePointer, offset: usize, data: &mut [u8]) {
        let begin = ptr.0 as usize + offset;
        let end = begin + data.len();
        data.copy_from_slice(&self.global_mem[begin..end]);
    }

    fn get_data_ptr<'a>(
        &'a mut self,
        space: StateSpace,
        state: &'a mut ThreadState,
    ) -> &'a mut [u8] {
        match space {
            StateSpace::Global => self.global_mem.as_mut_slice(),
            StateSpace::Stack => state.stack_data.as_mut_slice(),
            StateSpace::Shared => todo!(),
        }
    }

    fn run_cta(
        &mut self,
        nctaid: (u32, u32, u32),
        ctaid: (u32, u32, u32),
        ntid: (u32, u32, u32),
        desc: FuncFrameDesc,
        init_stack: &[u8],
    ) -> Result<(), VmError> {
        for x in 0..ntid.0 {
            for y in 0..ntid.1 {
                for z in 0..ntid.2 {
                    self.run_thread(nctaid, ctaid, ntid, (x, y, z), desc, init_stack)?;
                }
            }
        }
        Ok(())
    }

    fn run_thread(
        &mut self,
        nctaid: (u32, u32, u32),
        ctaid: (u32, u32, u32),
        ntid: (u32, u32, u32),
        tid: (u32, u32, u32),
        desc: FuncFrameDesc,
        init_stack: &[u8],
    ) -> Result<(), VmError> {
        let mut state = ThreadState::new();
        state.stack_data.extend_from_slice(&init_stack);
        state.frame_setup(desc);

        while state.num_frames() > 0 {
            let inst = self.fetch_instr(state.iptr_fetch_incr());
            match inst {
                Instruction::Load(space, dst, addr) => {
                    let addr = state.resolve_address(addr);
                    let data = match space {
                        StateSpace::Global => self.global_mem.as_slice(),
                        StateSpace::Stack => state.stack_data.as_slice(),
                        StateSpace::Shared => todo!(),
                    };
                    match dst {
                        RegOperand::B8(r) => state.set_b8(r, data[addr..addr + 1].try_into()?),
                        RegOperand::B16(r) => state.set_b16(r, data[addr..addr + 2].try_into()?),
                        RegOperand::B32(r) => state.set_b32(r, data[addr..addr + 4].try_into()?),
                        RegOperand::B64(r) => state.set_b64(r, data[addr..addr + 8].try_into()?),
                        RegOperand::B128(r) => state.set_b128(r, data[addr..addr + 16].try_into()?),
                        o => return Err(VmError::InvalidOperand(inst, o)),
                    }
                }
                Instruction::Store(space, src, addr) => {
                    let addr = state.resolve_address(addr);
                    match src {
                        RegOperand::B8(r) => {
                            let val = state.get_b8(r);
                            self.get_data_ptr(space, &mut state)[addr..addr + 1]
                                .copy_from_slice(&val);
                        }
                        RegOperand::B16(r) => {
                            let val = state.get_b16(r);
                            self.get_data_ptr(space, &mut state)[addr..addr + 2]
                                .copy_from_slice(&val);
                        }
                        RegOperand::B32(r) => {
                            let val = state.get_b32(r);
                            self.get_data_ptr(space, &mut state)[addr..addr + 4]
                                .copy_from_slice(&val);
                        }
                        RegOperand::B64(r) => {
                            let val = state.get_b64(r);
                            self.get_data_ptr(space, &mut state)[addr..addr + 8]
                                .copy_from_slice(&val);
                        }
                        RegOperand::B128(r) => {
                            let val = state.get_b128(r);
                            self.get_data_ptr(space, &mut state)[addr..addr + 16]
                                .copy_from_slice(&val);
                        }
                        o => return Err(VmError::InvalidOperand(inst, o)),
                    }
                }
                Instruction::Add(ty, dst, a, b) => {
                    use RegOperand::*;
                    match (dst, a, b) {
                        (B64(dst), B64(a), B64(b)) => match ty {
                            Type::U64 | Type::B64 => {
                                state.set_u64(dst, state.get_u64(a) + state.get_u64(b));
                            }
                            Type::S64 => {
                                state.set_i64(dst, state.get_i64(a) + state.get_i64(b));
                            }
                            _ => todo!(),
                        },
                        (B32(dst), B32(a), B32(b)) => match ty {
                            Type::U32 | Type::B32 => {
                                state.set_u32(dst, state.get_u32(a) + state.get_u32(b));
                            }
                            Type::S32 => {
                                state.set_i32(dst, state.get_i32(a) + state.get_i32(b));
                            }
                            Type::F32 => {
                                state.set_f32(dst, state.get_f32(a) + state.get_f32(b));
                            }
                            _ => todo!(),
                        },
                        _ => todo!(),
                    }
                }
                Instruction::Mul(ty, mode, dst, a, b) => {
                    use RegOperand::*;
                    match (mode, dst, a, b) {
                        (MulMode::Low, B64(dst), B64(a), B64(b)) => match ty {
                            Type::U64 | Type::B64 => {
                                state.set_u64(dst, state.get_u64(a) * state.get_u64(b));
                            }
                            _ => todo!(),
                        },
                        (MulMode::Low, B32(dst), B32(a), B32(b)) => match ty {
                            Type::U32 | Type::B32 => {
                                state.set_u32(dst, state.get_u32(a) * state.get_u32(b));
                            }
                            Type::S32 => {
                                state.set_i32(dst, state.get_i32(a) * state.get_i32(b));
                            }
                            _ => todo!(),
                        },
                        (MulMode::Wide, B64(dst), B32(a), B32(b)) => match ty {
                            Type::U32 => {
                                state.set_u64(
                                    dst,
                                    state.get_u32(a) as u64 * state.get_u32(b) as u64,
                                );
                            }
                            _ => todo!(),
                        },
                        _ => todo!(),
                    }
                }
                Instruction::ShiftLeft(ty, dst, a, b) => {
                    use RegOperand::*;
                    match (dst, a, b) {
                        (B64(dst), B64(a), B64(b)) => match ty {
                            Type::U64 | Type::B64 => {
                                state.set_u64(dst, state.get_u64(a) << state.get_u64(b));
                            }
                            _ => todo!(),
                        },
                        _ => todo!(),
                    }
                }
                Instruction::Convert {
                    dst_type,
                    src_type,
                    dst,
                    src,
                } => {
                    use RegOperand::*;
                    match (dst_type, src_type, dst, src) {
                        (Type::U64, Type::U32, B64(dst), B32(src)) => {
                            state.set_u64(dst, state.get_u32(src) as u64);
                        }
                        _ => todo!(),
                    }
                }
                Instruction::Move(_, dst, src) => {
                    use RegOperand::*;
                    match (dst, src) {
                        (B64(dst), B64(src)) => {
                            state.set_b64(dst, state.get_b64(src));
                        }
                        (B32(dst), Special(SpecialReg::ThreadIdX)) => {
                            state.set_u32(dst, tid.0);
                        }
                        (B32(dst), Special(SpecialReg::NumThreadX)) => {
                            state.set_u32(dst, ntid.0);
                        }
                        (B32(dst), Special(SpecialReg::CtaIdX)) => {
                            state.set_u32(dst, ctaid.0);
                        }
                        (B32(dst), Special(SpecialReg::NumCtaX)) => {
                            state.set_u32(dst, nctaid.0);
                        }
                        _ => todo!(),
                    }
                }
                Instruction::Const(dst, value) => {
                    use RegOperand::*;
                    match (dst, value) {
                        (B64(dst), Constant::U64(value)) => state.set_u64(dst, value),
                        (B32(dst), Constant::U32(value)) => state.set_u32(dst, value),
                        _ => todo!(),
                    }
                }
                Instruction::SetPredicate(ty, op, dst, a, b) => match (ty, a, b) {
                    (Type::U64, RegOperand::B64(a), RegOperand::B64(b)) => {
                        let a = u64::from_ne_bytes(state.get_b64(a));
                        let b = u64::from_ne_bytes(state.get_b64(b));
                        let value = match op {
                            PredicateOp::LessThan => a < b,
                            PredicateOp::LessThanEqual => a <= b,
                            PredicateOp::Equal => a == b,
                            PredicateOp::NotEqual => a != b,
                            PredicateOp::GreaterThan => a > b,
                            PredicateOp::GreaterThanEqual => a >= b,
                        };
                        state.set_pred(dst, value);
                    }
                    _ => todo!(),
                },
                Instruction::Jump { offset } => {
                    state.iptr.0 = (state.iptr.0 as isize + offset - 1) as usize;
                }
                Instruction::SkipIf(cond, expected) => {
                    if state.get_pred(cond) == expected {
                        state.iptr.0 += 1;
                    }
                }
                Instruction::Return => state.frame_teardown(),
            }
        }
        Ok(())
    }

    pub fn run(&mut self, params: LaunchParams, args: &[Argument]) -> Result<(), VmError> {
        let desc = self.descriptors[params.func_id];
        let arg_size: usize = args
            .iter()
            .map(|arg| match arg {
                Argument::Ptr(_) | Argument::U64(_) => std::mem::size_of::<u64>(),
                Argument::U32(_) => std::mem::size_of::<u32>(),
                Argument::Bytes(v) => v.len(),
            })
            .sum();
        if arg_size != desc.arg_size {
            return Err(VmError::ParamDataSizeMismatch);
        }

        let mut init_stack = Vec::with_capacity(desc.arg_size);
        for arg in args {
            match arg {
                Argument::Ptr(ptr) => {
                    let ptr_bytes = ptr.0.to_ne_bytes();
                    init_stack.extend_from_slice(&ptr_bytes);
                }
                Argument::U64(v) => {
                    let v_bytes = v.to_ne_bytes();
                    init_stack.extend_from_slice(&v_bytes);
                }
                Argument::U32(v) => {
                    let v_bytes = v.to_ne_bytes();
                    init_stack.extend_from_slice(&v_bytes);
                }
                Argument::Bytes(v) => {
                    init_stack.extend_from_slice(v);
                }
            }
        }
        for x in 0..params.grid_dim.0 {
            for y in 0..params.grid_dim.1 {
                for z in 0..params.grid_dim.2 {
                    self.run_cta(
                        params.grid_dim,
                        (x, y, z),
                        params.block_dim,
                        desc,
                        &init_stack,
                    )?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn simple() {
        let prog = vec![
            // load arguments into registers
            Instruction::Load(
                StateSpace::Stack,
                RegOperand::B64(Reg64 { id: 0 }),
                AddrOperand::StackRelative(-24),
            ),
            Instruction::Load(
                StateSpace::Stack,
                RegOperand::B64(Reg64 { id: 1 }),
                AddrOperand::StackRelative(-16),
            ),
            Instruction::Load(
                StateSpace::Stack,
                RegOperand::B64(Reg64 { id: 2 }),
                AddrOperand::StackRelative(-8),
            ),
            // load values from memory (pointed to by arguments)
            Instruction::Load(
                StateSpace::Global,
                RegOperand::B64(Reg64 { id: 3 }),
                AddrOperand::RegisterRelative(Reg64 { id: 0 }, 0),
            ),
            Instruction::Load(
                StateSpace::Global,
                RegOperand::B64(Reg64 { id: 4 }),
                AddrOperand::RegisterRelative(Reg64 { id: 1 }, 0),
            ),
            // add values
            Instruction::Add(
                Type::U64,
                RegOperand::B64(Reg64 { id: 5 }),
                RegOperand::B64(Reg64 { id: 3 }),
                RegOperand::B64(Reg64 { id: 4 }),
            ),
            // store result
            Instruction::Store(
                StateSpace::Global,
                RegOperand::B64(Reg64 { id: 5 }),
                AddrOperand::RegisterRelative(Reg64 { id: 2 }, 0),
            ),
            Instruction::Return,
        ];
        let desc = vec![FuncFrameDesc {
            iptr: IPtr(0),
            frame_size: 0,
            arg_size: 24,
            regs: RegDesc {
                b64_count: 6,
                ..Default::default()
            },
        }];
        const ALIGN: usize = std::mem::align_of::<u64>();
        let mut ctx = Context::new_raw(prog, desc);
        let a = ctx.alloc(8, ALIGN);
        let b = ctx.alloc(8, ALIGN);
        let c = ctx.alloc(8, ALIGN);
        ctx.write(a, 0, &1u64.to_ne_bytes());
        ctx.write(b, 0, &2u64.to_ne_bytes());
        ctx.run(
            LaunchParams::new().func(0).grid1d(1).block1d(1),
            &[Argument::Ptr(a), Argument::Ptr(b), Argument::Ptr(c)],
        )
        .unwrap();
        let mut res = [0u8; 8];
        ctx.read(c, 0, &mut res);
        assert_eq!(u64::from_ne_bytes(res), 3);
    }

    #[test]
    fn multiple_threads() {
        let prog = vec![
            // load arguments into registers
            Instruction::Load(
                StateSpace::Stack,
                RegOperand::B64(Reg64 { id: 0 }),
                AddrOperand::StackRelative(-24),
            ),
            Instruction::Load(
                StateSpace::Stack,
                RegOperand::B64(Reg64 { id: 1 }),
                AddrOperand::StackRelative(-16),
            ),
            Instruction::Load(
                StateSpace::Stack,
                RegOperand::B64(Reg64 { id: 2 }),
                AddrOperand::StackRelative(-8),
            ),
            // load thread index
            Instruction::Move(
                Type::U32,
                RegOperand::B32(Reg32 { id: 0 }),
                RegOperand::Special(SpecialReg::ThreadIdX),
            ),
            Instruction::Convert {
                dst_type: Type::U64,
                src_type: Type::U32,
                dst: RegOperand::B64(Reg64 { id: 6 }),
                src: RegOperand::B32(Reg32 { id: 0 }),
            },
            // multiply thread index by 8 (size of u64)
            Instruction::Const(RegOperand::B64(Reg64 { id: 7 }), Constant::U64(8)),
            Instruction::Mul(
                Type::U64,
                MulMode::Low,
                RegOperand::B64(Reg64 { id: 6 }),
                RegOperand::B64(Reg64 { id: 6 }),
                RegOperand::B64(Reg64 { id: 7 }),
            ),
            // offset argument pointers by thread index
            Instruction::Add(
                Type::U64,
                RegOperand::B64(Reg64 { id: 0 }),
                RegOperand::B64(Reg64 { id: 0 }),
                RegOperand::B64(Reg64 { id: 6 }),
            ),
            Instruction::Add(
                Type::U64,
                RegOperand::B64(Reg64 { id: 1 }),
                RegOperand::B64(Reg64 { id: 1 }),
                RegOperand::B64(Reg64 { id: 6 }),
            ),
            Instruction::Add(
                Type::U64,
                RegOperand::B64(Reg64 { id: 2 }),
                RegOperand::B64(Reg64 { id: 2 }),
                RegOperand::B64(Reg64 { id: 6 }),
            ),
            // load values from memory (pointed to by offset arguments)
            Instruction::Load(
                StateSpace::Global,
                RegOperand::B64(Reg64 { id: 3 }),
                AddrOperand::RegisterRelative(Reg64 { id: 0 }, 0),
            ),
            Instruction::Load(
                StateSpace::Global,
                RegOperand::B64(Reg64 { id: 4 }),
                AddrOperand::RegisterRelative(Reg64 { id: 1 }, 0),
            ),
            // add values
            Instruction::Add(
                Type::U64,
                RegOperand::B64(Reg64 { id: 5 }),
                RegOperand::B64(Reg64 { id: 3 }),
                RegOperand::B64(Reg64 { id: 4 }),
            ),
            // store result
            Instruction::Store(
                StateSpace::Global,
                RegOperand::B64(Reg64 { id: 5 }),
                AddrOperand::RegisterRelative(Reg64 { id: 2 }, 0),
            ),
            Instruction::Return,
        ];
        let desc = vec![FuncFrameDesc {
            iptr: IPtr(0),
            frame_size: 0,
            arg_size: 24,
            regs: RegDesc {
                b32_count: 1,
                b64_count: 8,
                ..Default::default()
            },
        }];

        const ALIGN: usize = std::mem::align_of::<u64>();
        const N: usize = 10;

        let mut ctx = Context::new_raw(prog, desc);
        let a = ctx.alloc(8 * N, ALIGN);
        let b = ctx.alloc(8 * N, ALIGN);
        let c = ctx.alloc(8 * N, ALIGN);

        let data_a = vec![1u64; N];
        let data_b = vec![2u64; N];
        ctx.write(a, 0, bytemuck::cast_slice(&data_a));
        ctx.write(b, 0, bytemuck::cast_slice(&data_b));

        ctx.run(
            LaunchParams::new().func(0).grid1d(1).block1d(N as u32),
            &[Argument::Ptr(a), Argument::Ptr(b), Argument::Ptr(c)],
        )
        .unwrap();

        let mut res = vec![0u64; N];
        ctx.read(c, 0, bytemuck::cast_slice_mut(&mut res));

        res.iter().for_each(|v| assert_eq!(*v, 3));
    }
}
