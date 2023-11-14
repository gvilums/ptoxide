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

// #[derive(Clone, Copy, Debug)]
// struct Value(u128);

// impl Value {
//     fn assemble(data:  &[u8]) -> Self {
//         let mut buf = [0u8; 16];
//         let len = data.len().min(16);
//         buf[..len].copy_from_slice(&data[..len]);
//         Value(u128::from_ne_bytes(buf))
//     }

//     fn as_u128(self) -> u128 {
//         self.0
//     }

//     fn as_u64(self) -> u64 {
//         self.0 as u64
//     }

//     fn as_u32(self) -> u32 {
//         self.0 as u32
//     }

//     fn as_u16(self) -> u16 {
//         self.0 as u16
//     }

//     fn as_u8(self) -> u8 {
//         self.0 as u8
//     }

//     fn as_b128(self) -> [u8; 16] {
//         self.0.to_ne_bytes()
//     }

//     fn as_b64(self) -> [u8; 8] {
//         self.as_u64().to_ne_bytes()
//     }

//     fn as_b32(self) -> [u8; 4] {
//         self.as_u32().to_ne_bytes()
//     }

//     fn as_b16(self) -> [u8; 2] {
//         self.as_u16().to_ne_bytes()
//     }

//     fn as_b8(self) -> [u8; 1] {
//         self.as_u8().to_ne_bytes()
//     }
// }


#[derive(Clone, Copy, Debug)]
pub enum Instruction {
    Load(StateSpace, RegOperand, RegOperand),
    Store(StateSpace, RegOperand, RegOperand),
    Convert {
        dst_type: Type,
        src_type: Type,
        dst: RegOperand,
        src: RegOperand,
    },
    Move(Type, RegOperand, RegOperand),
    Const(RegOperand, Constant),
    Add(Type, RegOperand, RegOperand, RegOperand),
    Sub(Type, RegOperand, RegOperand, RegOperand),
    Or(Type, RegOperand, RegOperand, RegOperand),
    And(Type, RegOperand, RegOperand, RegOperand),
    Mul(Type, MulMode, RegOperand, RegOperand, RegOperand),
    // todo this should just be expressed as sub with 0
    Neg(Type, RegOperand, RegOperand),
    ShiftLeft(Type, RegOperand, RegOperand, RegOperand),
    SetPredicate(Type, PredicateOp, RegPred, RegOperand, RegOperand),
    BarrierSync {
        idx: RegOperand,
        cnt: Option<RegOperand>,
    },
    BarrierArrive {
        idx: RegOperand,
        cnt: Option<RegOperand>,
    },
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

#[derive(Debug, Clone)]
struct ThreadState {
    iptr: IPtr,
    regs: Vec<Registers>,
    stack_data: Vec<u8>,
    stack_frames: Vec<FrameMeta>,
    nctaid: (u32, u32, u32),
    ctaid: (u32, u32, u32),
    ntid: (u32, u32, u32),
    tid: (u32, u32, u32),
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
    pub shared_size: usize,
}

#[derive(Debug)]
pub struct Context {
    global_mem: Vec<u8>,
    instructions: Vec<Instruction>,
    descriptors: Vec<FuncFrameDesc>,
    symbol_map: HashMap<String, Symbol>,
}

#[derive(Debug, Clone)]
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

macro_rules! byte_reg_funcs {
    ($($t:ty, $get:ident, $set:ident, $field:ident, $n:expr);*) => {
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

macro_rules! int_reg_funcs {
    ($($t:ty, $get:ident, $set:ident, $field:ident, $t2:ty);*) => {
        $(
            #[allow(dead_code)]
            fn $get(&self, reg: $t) -> $t2 {
                <$t2>::from_ne_bytes(self.regs.last().unwrap().$field[reg.id])
            }

            #[allow(dead_code)]
            fn $set(&mut self, reg: $t, value: $t2) {
                self.regs.last_mut().unwrap().$field[reg.id] = value.to_ne_bytes();
            }
        )*
    };
}

impl ThreadState {
    fn new(
        nctaid: (u32, u32, u32),
        ctaid: (u32, u32, u32),
        ntid: (u32, u32, u32),
        tid: (u32, u32, u32),
    ) -> ThreadState {
        ThreadState {
            iptr: IPtr(0),
            regs: Vec::new(),
            stack_data: Vec::new(),
            stack_frames: Vec::new(),
            nctaid,
            ctaid,
            ntid,
            tid,
        }
    }

    fn get_pred(&self, reg: RegPred) -> bool {
        self.regs.last().unwrap().pred[reg.id]
    }

    fn set_pred(&mut self, reg: RegPred, value: bool) {
        self.regs.last_mut().unwrap().pred[reg.id] = value;
    }

    byte_reg_funcs!(
        Reg8, get_b8, set_b8, b8, 1;
        Reg16, get_b16, set_b16, b16, 2;
        Reg32, get_b32, set_b32, b32, 4;
        Reg64, get_b64, set_b64, b64, 8;
        Reg128, get_b128, set_b128, b128, 16
    );

    int_reg_funcs!(
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
        let frame_meta = self.stack_frames.pop().unwrap();
        self.stack_data
            .truncate(self.stack_data.len() - frame_meta.frame_size);
        self.regs.pop();
        self.iptr = frame_meta.return_addr;
    }

    fn frame_setup(&mut self, desc: FuncFrameDesc) {
        self.stack_frames.push(FrameMeta {
            return_addr: self.iptr,
            frame_size: desc.frame_size,
        });
        self.stack_data
            .resize(self.stack_data.len() + desc.frame_size, 0);
        self.regs.push(Registers::new(desc.regs));
        self.iptr = desc.iptr;
    }

    fn read_reg_signed(&self, reg: RegOperand) -> VmResult<isize> {
        match reg {
            RegOperand::Pred(_) | RegOperand::Special(_) => Err(VmError::InvalidAddressOperandRegister(reg)),
            RegOperand::B8(r) => Ok(self.get_i8(r) as isize),
            RegOperand::B16(r) => Ok(self.get_i16(r) as isize),
            RegOperand::B32(r) => Ok(self.get_i32(r) as isize),
            RegOperand::B64(r) => Ok(self.get_i64(r) as isize),
            RegOperand::B128(r) => Ok(self.get_i128(r) as isize),
        }
    }

    fn read_reg_unsigned(&self, reg: RegOperand) -> VmResult<usize> {
        match reg {
            RegOperand::Pred(_) | RegOperand::Special(_) => Err(VmError::InvalidAddressOperandRegister(reg)),
            RegOperand::B8(r) => Ok(self.get_i8(r) as usize),
            RegOperand::B16(r) => Ok(self.get_i16(r) as usize),
            RegOperand::B32(r) => Ok(self.get_i32(r) as usize),
            RegOperand::B64(r) => Ok(self.get_i64(r) as usize),
            RegOperand::B128(r) => Ok(self.get_i128(r) as usize),
        }
    }

    fn get_stack_ptr(&self) -> VmResult<usize> {
        let frame_size = self.stack_frames.last().unwrap().frame_size;
        Ok(self.stack_data.len() - frame_size)
    }
}

#[derive(thiserror::Error, Debug)]
pub enum VmError {
    #[error("invalid address operand register {0:?}")]
    InvalidAddressOperandRegister(RegOperand),
    #[error("Invalid register operand {:?} for instruction {:?}", .1, .0)]
    InvalidOperand(Instruction, RegOperand),
    #[error("Parameter data did not match descriptor")]
    ParamDataSizeMismatch,
    #[error("Invalid function id {0}")]
    InvalidFunctionId(usize),
    #[error("Invalid function name {0}")]
    InvalidFunctionName(String),
    #[error("Slice size mismatch")]
    SliceSizeMismatch(#[from] std::array::TryFromSliceError),
    #[error("Parse error: {0:?}")]
    ParseError(#[from] crate::ast::ParseErr),
    #[error("Compile error: {0:?}")]
    CompileError(#[from] crate::compiler::CompilationError),
}

type VmResult<T> = Result<T, VmError>;

#[derive(Clone, Copy, Debug)]
pub struct DevicePointer(u64);

pub enum Argument<'a> {
    Ptr(DevicePointer),
    U64(u64),
    U32(u32),
    Bytes(&'a [u8]),
}

#[derive(Clone, Copy, Debug)]
enum FuncIdent<'a> {
    Name(&'a str),
    Id(usize),
}

#[derive(Clone, Copy, Debug)]
pub struct LaunchParams<'a> {
    func_id: FuncIdent<'a>,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
}

enum ThreadResult {
    Continue,
    Sync(usize, Option<usize>),
    Arrive(usize, Option<usize>),
    Exit,
}

impl<'a> LaunchParams<'a> {

    pub fn func(name: &'a str) -> LaunchParams<'a> {
        LaunchParams {
            func_id: FuncIdent::Name(name),
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
        }
    }

    pub fn func_id(id: usize) -> LaunchParams<'a> {
        LaunchParams {
            func_id: FuncIdent::Id(id),
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
        }
    }

    pub fn grid1d(mut self, x: u32) -> LaunchParams<'a> {
        self.grid_dim = (x, 1, 1);
        self
    }

    pub fn grid2d(mut self, x: u32, y: u32) -> LaunchParams<'a> {
        self.grid_dim = (x, y, 1);
        self
    }

    pub fn block1d(mut self, x: u32) -> LaunchParams<'a> {
        self.block_dim = (x, 1, 1);
        self
    }

    pub fn block2d(mut self, x: u32, y: u32) -> LaunchParams<'a> {
        self.block_dim = (x, y, 1);
        self
    }
}

#[derive(Clone, Debug)]
struct Barrier {
    target: usize,
    arrived: usize,
    blocked: Vec<ThreadState>,
}

struct Barriers {
    barriers: Vec<Option<Barrier>>
}

impl Barriers {
    pub fn new() -> Self {
        Barriers {
            barriers: Vec::new()
        }
    }

    pub fn arrive(&mut self, idx: usize, target: usize) -> VmResult<Vec<ThreadState>> {
        todo!()
    }

    pub fn block(&mut self, idx: usize, target: usize, thread: ThreadState) -> VmResult<Vec<ThreadState>> {
        self.assert_size(idx);
        if let Some(ref mut barr) = self.barriers[idx] {
            barr.blocked.push(thread);
            barr.arrived += 1;
            if barr.arrived == barr.target {
                let barr = self.barriers[idx].take().unwrap();
                return Ok(barr.blocked);
            } 
        } else {
            self.barriers[idx] = Some(Barrier {
                target,
                arrived: 1,
                blocked: vec![thread],
            });
        }
        Ok(Vec::new())
    }

    fn assert_size(&mut self, idx: usize) {
        if idx >= self.barriers.len() {
            self.barriers.resize(idx + 1, None);
        }
    }
}

macro_rules! binary_op {
    ($threadop:expr, $tyop:expr, $dstop:expr, $aop:expr, $bop:expr; 
        $($target_ty:pat, $reg_size:ident, $op:ident, $getter:ident, $setter:ident);*$(;)?) => {
        match ($tyop, $dstop, $aop, $bop) {
        $(
            ($target_ty, RegOperand::$reg_size(dst), RegOperand::$reg_size(a), RegOperand::$reg_size(c)) => {
                let val = $threadop.$getter(a).$op($threadop.$getter(c));
                $threadop.$setter(dst, val);
            }
        )*
            _ => todo!()
        }
    };
}

macro_rules! unary_op {
    ($threadop:expr, $tyop:expr, $dstop:expr, $srcop:expr; 
        $($target_ty:pat, $reg_size:ident, $op:ident, $getter:ident, $setter:ident);*$(;)?) => {
        match ($tyop, $dstop, $srcop) {
        $(
            ($target_ty, RegOperand::$reg_size(dst), RegOperand::$reg_size(src)) => {
                let val = $threadop.$getter(src).$op();
                $threadop.$setter(dst, val);
            }
        )*
            _ => todo!()
        }
    };
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

    pub fn new_with_module(module: &str) -> VmResult<Self> {
        let module = crate::ast::parse_program(module)?;
        let compiled = crate::compiler::compile(module)?;
        Ok(Self {
            global_mem: Vec::new(),
            instructions: compiled.instructions,
            descriptors: compiled.func_descriptors,
            symbol_map: compiled.symbol_map,
        })
    }

    pub fn load(&mut self, module: &str) -> VmResult<()> {
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
        shared: &'a mut [u8],
    ) -> &'a mut [u8] {
        match space {
            StateSpace::Global => self.global_mem.as_mut_slice(),
            StateSpace::Stack => state.stack_data.as_mut_slice(),
            StateSpace::Shared => shared,
        }
    }

    fn run_cta(
        &mut self,
        nctaid: (u32, u32, u32),
        ctaid: (u32, u32, u32),
        ntid: (u32, u32, u32),
        desc: FuncFrameDesc,
        init_stack: &[u8],
    ) -> VmResult<()> {
        let mut shared_mem = vec![0u8; desc.shared_size];

        let mut runnable = Vec::new();
        for x in 0..ntid.0 {
            for y in 0..ntid.1 {
                for z in 0..ntid.2 {
                    let mut state = ThreadState::new(nctaid, ctaid, ntid, (x, y, z));
                    state.stack_data.extend_from_slice(&init_stack);
                    state.frame_setup(desc);
                    runnable.push(state);
                }
            }
        }
        let cta_size = (ntid.0 * ntid.1 * ntid.2) as usize;

        let mut barriers = Barriers::new();

        while let Some(mut state) = runnable.pop() {
            loop {
                match self.step_thread(&mut state, &mut shared_mem)? {
                    ThreadResult::Continue => continue, 
                    ThreadResult::Arrive(idx, cnt) => {
                        let cnt = cnt.unwrap_or(cta_size);
                        runnable.extend(barriers.arrive(idx, cnt)?);
                        continue
                    },
                    ThreadResult::Sync(idx, cnt) => {
                        let cnt = cnt.unwrap_or(cta_size);
                        runnable.extend(barriers.block(idx, cnt, state)?);
                        break
                    },
                    ThreadResult::Exit => break
                }
            }
        }
        Ok(())
    }

    fn step_thread(
        &mut self,
        thread: &mut ThreadState,
        shared_mem: &mut [u8],
    ) -> VmResult<ThreadResult> {
        let inst = self.fetch_instr(thread.iptr_fetch_incr());
        match inst {
            Instruction::Load(space, dst, addr) => {
                // let addr = thread.resolve_address(addr)?;
                let addr = thread.read_reg_unsigned(addr)?;
                let data = match space {
                    StateSpace::Global => self.global_mem.as_slice(),
                    StateSpace::Stack => thread.stack_data.as_slice(),
                    StateSpace::Shared => shared_mem,
                };
                match dst {
                    RegOperand::B8(r) => thread.set_b8(r, data[addr..addr + 1].try_into()?),
                    RegOperand::B16(r) => thread.set_b16(r, data[addr..addr + 2].try_into()?),
                    RegOperand::B32(r) => thread.set_b32(r, data[addr..addr + 4].try_into()?),
                    RegOperand::B64(r) => thread.set_b64(r, data[addr..addr + 8].try_into()?),
                    RegOperand::B128(r) => thread.set_b128(r, data[addr..addr + 16].try_into()?),
                    o => return Err(VmError::InvalidOperand(inst, o)),
                }
            }
            Instruction::Store(space, src, addr) => {
                // let addr = thread.resolve_address(addr)?;
                let addr = thread.read_reg_unsigned(addr)?;
                match src {
                    RegOperand::B8(r) => {
                        let val = thread.get_b8(r);
                        self.get_data_ptr(space, thread, shared_mem)[addr..addr + 1].copy_from_slice(&val);
                    }
                    RegOperand::B16(r) => {
                        let val = thread.get_b16(r);
                        self.get_data_ptr(space, thread, shared_mem)[addr..addr + 2].copy_from_slice(&val);
                    }
                    RegOperand::B32(r) => {
                        let val = thread.get_b32(r);
                        self.get_data_ptr(space, thread, shared_mem)[addr..addr + 4].copy_from_slice(&val);
                    }
                    RegOperand::B64(r) => {
                        let val = thread.get_b64(r);
                        self.get_data_ptr(space, thread, shared_mem)[addr..addr + 8].copy_from_slice(&val);
                    }
                    RegOperand::B128(r) => {
                        let val = thread.get_b128(r);
                        self.get_data_ptr(space, thread, shared_mem)[addr..addr + 16].copy_from_slice(&val);
                    }
                    o => return Err(VmError::InvalidOperand(inst, o)),
                }
            }
            Instruction::Sub(ty, dst, a, b) => {
                use std::ops::Sub;
                binary_op! {
                    thread, ty, dst, a, b;
                    Type::U64 | Type::B64, B64, sub, get_u64, set_u64;
                    Type::S64, B64, sub, get_i64, set_i64;
                    Type::U32 | Type::B32, B32, sub, get_u32, set_u32;
                    Type::S32, B32, sub, get_i32, set_i32;
                    Type::F32, B32, sub, get_f32, set_f32;
                };
            }
            Instruction::Add(ty, dst, a, b) => {
                use std::ops::Add;
                binary_op! {
                    thread, ty, dst, a, b;
                    Type::U64 | Type::B64, B64, add, get_u64, set_u64;
                    Type::S64, B64, add, get_i64, set_i64;
                    Type::U32 | Type::B32, B32, add, get_u32, set_u32;
                    Type::S32, B32, add, get_i32, set_i32;
                    Type::F64, B64, add, get_f64, set_f64;
                    Type::F32, B32, add, get_f32, set_f32;
                };
            }
            Instruction::Mul(ty, mode, dst, a, b) => {
                use std::ops::Mul;
                match mode {
                    MulMode::Low => {
                        binary_op! {
                            thread, ty, dst, a, b;
                            Type::U64 | Type::B64, B64, mul, get_u64, set_u64;
                            Type::S64, B64, mul, get_i64, set_i64;
                            Type::U32 | Type::B32, B32, mul, get_u32, set_u32;
                            Type::S32, B32, mul, get_i32, set_i32;
                            Type::F32, B32, mul, get_f32, set_f32;
                        }
                    },
                    MulMode::High => todo!(),
                    MulMode::Wide => {
                        use RegOperand::*;
                        match (ty, dst, a, b) {
                            (Type::U32, B64(dst), B32(a), B32(b)) => {
                                thread
                                    .set_u64(dst, thread.get_u32(a) as u64 * thread.get_u32(b) as u64);
                            }
                            (Type::S32, B64(dst), B32(a), B32(b)) => {
                                thread
                                    .set_i64(dst, thread.get_i32(a) as i64 * thread.get_i32(b) as i64);
                            }
                            _ => todo!()
                        }
                    },
                }
            }
            Instruction::Or(ty, dst, a, b) => {
                use std::ops::BitOr;
                binary_op! {
                    thread, ty, dst, a, b;
                    Type::Pred, Pred, bitor, get_pred, set_pred;
                    Type::U64 | Type::B64 | Type::S64, B64, bitor, get_u64, set_u64;
                    Type::U32 | Type::B32 | Type::S32, B32, bitor, get_u32, set_u32;
                };
            }
            Instruction::And(ty, dst, a, b) => {
                use std::ops::BitAnd;
                binary_op! {
                    thread, ty, dst, a, b;
                    Type::Pred, Pred, bitand, get_pred, set_pred;
                    Type::U64 | Type::B64 | Type::S64, B64, bitand, get_u64, set_u64;
                    Type::U32 | Type::B32 | Type::S64, B32, bitand, get_u32, set_u32;
                };
            }
            Instruction::Neg(ty, dst, src) => {
                use std::ops::Neg;
                unary_op! {
                    thread, ty, dst, src;
                    Type::S64, B64, neg, get_i64, set_i64;
                    Type::S32, B32, neg, get_i32, set_i32;
                    Type::F32, B32, neg, get_f32, set_f32;
                };
            }
            Instruction::ShiftLeft(ty, dst, a, b) => {
                use std::ops::Shl;
                binary_op! {
                    thread, ty, dst, a, b;
                    Type::B64, B64, shl, get_u64, set_u64;
                    Type::B32, B32, shl, get_u32, set_u32;
                    Type::B16, B16, shl, get_u16, set_u16;
                };
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
                        thread.set_u64(dst, thread.get_u32(src) as u64);
                    }
                    _ => todo!(),
                }
            }
            Instruction::Move(_, dst, src) => {
                use RegOperand::*;
                match (dst, src) {
                    (B64(dst), B64(src)) => {
                        thread.set_b64(dst, thread.get_b64(src));
                    }
                    (B32(dst), B32(src)) => {
                        thread.set_b32(dst, thread.get_b32(src));
                    }
                    (B32(dst), Special(SpecialReg::ThreadIdX)) => {
                        thread.set_u32(dst, thread.tid.0);
                    }
                    (B32(dst), Special(SpecialReg::NumThreadX)) => {
                        thread.set_u32(dst, thread.ntid.0);
                    }
                    (B32(dst), Special(SpecialReg::CtaIdX)) => {
                        thread.set_u32(dst, thread.ctaid.0);
                    }
                    (B32(dst), Special(SpecialReg::NumCtaX)) => {
                        thread.set_u32(dst, thread.nctaid.0);
                    }
                    (B32(dst), Special(SpecialReg::ThreadIdY)) => {
                        thread.set_u32(dst, thread.tid.1);
                    }
                    (B32(dst), Special(SpecialReg::NumThreadY)) => {
                        thread.set_u32(dst, thread.ntid.1);
                    }
                    (B32(dst), Special(SpecialReg::CtaIdY)) => {
                        thread.set_u32(dst, thread.ctaid.1);
                    }
                    (B32(dst), Special(SpecialReg::NumCtaY)) => {
                        thread.set_u32(dst, thread.nctaid.1);
                    }
                    (B64(dst), Special(SpecialReg::StackPtr)) => {
                        thread.set_u64(dst, thread.get_stack_ptr()? as u64)
                    }
                    _ => todo!(),
                }
            }
            Instruction::Const(dst, value) => {
                use RegOperand::*;
                match (dst, value) {
                    (B64(dst), Constant::U64(value)) => thread.set_u64(dst, value),
                    (B64(dst), Constant::S64(value)) => thread.set_i64(dst, value),
                    (B32(dst), Constant::U32(value)) => thread.set_u32(dst, value),
                    (B32(dst), Constant::S32(value)) => thread.set_i32(dst, value),
                    (B32(dst), Constant::F32(value)) => thread.set_f32(dst, value),

                    // temporary fix for operations that move address of local into register
                    (B32(dst), Constant::U64(value)) => thread.set_u32(dst, value as u32),
                    _ => todo!(),
                }
            }
            Instruction::SetPredicate(ty, op, dst, a, b) => match (ty, a, b) {
                (Type::U64, RegOperand::B64(a), RegOperand::B64(b)) => {
                    let a = thread.get_u64(a);
                    let b = thread.get_u64(b);
                    let value = match op {
                        PredicateOp::LessThan => a < b,
                        PredicateOp::LessThanEqual => a <= b,
                        PredicateOp::Equal => a == b,
                        PredicateOp::NotEqual => a != b,
                        PredicateOp::GreaterThan => a > b,
                        PredicateOp::GreaterThanEqual => a >= b,
                    };
                    thread.set_pred(dst, value);
                }
                (Type::S64, RegOperand::B64(a), RegOperand::B64(b)) => {
                    let a = thread.get_i64(a);
                    let b = thread.get_i64(b);
                    let value = match op {
                        PredicateOp::LessThan => a < b,
                        PredicateOp::LessThanEqual => a <= b,
                        PredicateOp::Equal => a == b,
                        PredicateOp::NotEqual => a != b,
                        PredicateOp::GreaterThan => a > b,
                        PredicateOp::GreaterThanEqual => a >= b,
                    };
                    thread.set_pred(dst, value);
                }
                _ => todo!()
            },
            Instruction::BarrierSync { idx, cnt } => {
                let RegOperand::B32(idx) = idx else {
                    return Err(VmError::InvalidOperand(inst, idx));
                };
                let idx = thread.get_u32(idx) as usize;
                let cnt = if let Some(cnt) = cnt {
                    let RegOperand::B32(cnt) = cnt else {
                        return Err(VmError::InvalidOperand(inst, cnt));
                    };
                    Some(thread.get_u32(cnt) as usize)
                } else {
                    None
                };
                return Ok(ThreadResult::Sync(idx, cnt));
            }
            Instruction::BarrierArrive { idx, cnt } => {
                let RegOperand::B32(idx) = idx else {
                    return Err(VmError::InvalidOperand(inst, idx));
                };
                let idx = thread.get_u32(idx) as usize;
                let cnt = if let Some(cnt) = cnt {
                    let RegOperand::B32(cnt) = cnt else {
                        return Err(VmError::InvalidOperand(inst, cnt));
                    };
                    Some(thread.get_u32(cnt) as usize)
                } else {
                    None
                };
                return Ok(ThreadResult::Arrive(idx, cnt));
            }
            Instruction::Jump { offset } => {
                thread.iptr.0 = (thread.iptr.0 as isize + offset - 1) as usize;
            }
            Instruction::SkipIf(cond, expected) => {
                if thread.get_pred(cond) == expected {
                    thread.iptr.0 += 1;
                }
            }
            Instruction::Return => thread.frame_teardown(),
        }
        if thread.stack_frames.is_empty() {
            Ok(ThreadResult::Exit)
        } else {
            Ok(ThreadResult::Continue)
        }
    }

    pub fn run(&mut self, params: LaunchParams, args: &[Argument]) -> VmResult<()> {
        let desc = match params.func_id {
            FuncIdent::Name(s) => {
                let Some(Symbol::Function(i)) = self.symbol_map.get(s) else {
                    return Err(VmError::InvalidFunctionName(s.to_string()));
                };
                self.descriptors[*i]
            },
            FuncIdent::Id(i) => *self.descriptors.get(i).ok_or(VmError::InvalidFunctionId(i))?,
        };
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

// #[cfg(test)]
// mod test {
//     use super::*;

//     #[test]
//     fn simple() {
//         let prog = vec![
//             // load arguments into registers
//             Instruction::Load(
//                 StateSpace::Stack,
//                 RegOperand::B64(Reg64 { id: 0 }),
//                 AddrOperand::StackRelative(-24),
//             ),
//             Instruction::Load(
//                 StateSpace::Stack,
//                 RegOperand::B64(Reg64 { id: 1 }),
//                 AddrOperand::StackRelative(-16),
//             ),
//             Instruction::Load(
//                 StateSpace::Stack,
//                 RegOperand::B64(Reg64 { id: 2 }),
//                 AddrOperand::StackRelative(-8),
//             ),
//             // load values from memory (pointed to by arguments)
//             Instruction::Load(
//                 StateSpace::Global,
//                 RegOperand::B64(Reg64 { id: 3 }),
//                 AddrOperand::RegisterRelative(RegOperand::B64(Reg64 { id: 0 }), 0),
//             ),
//             Instruction::Load(
//                 StateSpace::Global,
//                 RegOperand::B64(Reg64 { id: 4 }),
//                 AddrOperand::RegisterRelative(RegOperand::B64(Reg64 { id: 1 }), 0),
//             ),
//             // add values
//             Instruction::Add(
//                 Type::U64,
//                 RegOperand::B64(Reg64 { id: 5 }),
//                 RegOperand::B64(Reg64 { id: 3 }),
//                 RegOperand::B64(Reg64 { id: 4 }),
//             ),
//             // store result
//             Instruction::Store(
//                 StateSpace::Global,
//                 RegOperand::B64(Reg64 { id: 5 }),
//                 AddrOperand::RegisterRelative(RegOperand::B64(Reg64 { id: 2 }), 0),
//             ),
//             Instruction::Return,
//         ];
//         let desc = vec![FuncFrameDesc {
//             iptr: IPtr(0),
//             frame_size: 0,
//             shared_size: 0,
//             arg_size: 24,
//             regs: RegDesc {
//                 b64_count: 6,
//                 ..Default::default()
//             },
//         }];
//         const ALIGN: usize = std::mem::align_of::<u64>();
//         let mut ctx = Context::new_raw(prog, desc);
//         let a = ctx.alloc(8, ALIGN);
//         let b = ctx.alloc(8, ALIGN);
//         let c = ctx.alloc(8, ALIGN);
//         ctx.write(a, 0, &1u64.to_ne_bytes());
//         ctx.write(b, 0, &2u64.to_ne_bytes());
//         ctx.run(
//             LaunchParams::func_id(0).grid1d(1).block1d(1),
//             &[Argument::Ptr(a), Argument::Ptr(b), Argument::Ptr(c)],
//         )
//         .unwrap();
//         let mut res = [0u8; 8];
//         ctx.read(c, 0, &mut res);
//         assert_eq!(u64::from_ne_bytes(res), 3);
//     }

//     #[test]
//     fn multiple_threads() {
//         let prog = vec![
//             // load arguments into registers
//             Instruction::Load(
//                 StateSpace::Stack,
//                 RegOperand::B64(Reg64 { id: 0 }),
//                 AddrOperand::StackRelative(-24),
//             ),
//             Instruction::Load(
//                 StateSpace::Stack,
//                 RegOperand::B64(Reg64 { id: 1 }),
//                 AddrOperand::StackRelative(-16),
//             ),
//             Instruction::Load(
//                 StateSpace::Stack,
//                 RegOperand::B64(Reg64 { id: 2 }),
//                 AddrOperand::StackRelative(-8),
//             ),
//             // load thread index
//             Instruction::Move(
//                 Type::U32,
//                 RegOperand::B32(Reg32 { id: 0 }),
//                 RegOperand::Special(SpecialReg::ThreadIdX),
//             ),
//             Instruction::Convert {
//                 dst_type: Type::U64,
//                 src_type: Type::U32,
//                 dst: RegOperand::B64(Reg64 { id: 6 }),
//                 src: RegOperand::B32(Reg32 { id: 0 }),
//             },
//             // multiply thread index by 8 (size of u64)
//             Instruction::Const(RegOperand::B64(Reg64 { id: 7 }), Constant::U64(8)),
//             Instruction::Mul(
//                 Type::U64,
//                 MulMode::Low,
//                 RegOperand::B64(Reg64 { id: 6 }),
//                 RegOperand::B64(Reg64 { id: 6 }),
//                 RegOperand::B64(Reg64 { id: 7 }),
//             ),
//             // offset argument pointers by thread index
//             Instruction::Add(
//                 Type::U64,
//                 RegOperand::B64(Reg64 { id: 0 }),
//                 RegOperand::B64(Reg64 { id: 0 }),
//                 RegOperand::B64(Reg64 { id: 6 }),
//             ),
//             Instruction::Add(
//                 Type::U64,
//                 RegOperand::B64(Reg64 { id: 1 }),
//                 RegOperand::B64(Reg64 { id: 1 }),
//                 RegOperand::B64(Reg64 { id: 6 }),
//             ),
//             Instruction::Add(
//                 Type::U64,
//                 RegOperand::B64(Reg64 { id: 2 }),
//                 RegOperand::B64(Reg64 { id: 2 }),
//                 RegOperand::B64(Reg64 { id: 6 }),
//             ),
//             // load values from memory (pointed to by offset arguments)
//             Instruction::Load(
//                 StateSpace::Global,
//                 RegOperand::B64(Reg64 { id: 3 }),
//                 AddrOperand::RegisterRelative(RegOperand::B64(Reg64 { id: 0 }), 0),
//             ),
//             Instruction::Load(
//                 StateSpace::Global,
//                 RegOperand::B64(Reg64 { id: 4 }),
//                 AddrOperand::RegisterRelative(RegOperand::B64(Reg64 { id: 1 }), 0),
//             ),
//             // add values
//             Instruction::Add(
//                 Type::U64,
//                 RegOperand::B64(Reg64 { id: 5 }),
//                 RegOperand::B64(Reg64 { id: 3 }),
//                 RegOperand::B64(Reg64 { id: 4 }),
//             ),
//             // store result
//             Instruction::Store(
//                 StateSpace::Global,
//                 RegOperand::B64(Reg64 { id: 5 }),
//                 AddrOperand::RegisterRelative(RegOperand::B64(Reg64 { id: 2 }), 0),
//             ),
//             Instruction::Return,
//         ];
//         let desc = vec![FuncFrameDesc {
//             iptr: IPtr(0),
//             frame_size: 0,
//             shared_size: 0,
//             arg_size: 24,
//             regs: RegDesc {
//                 b32_count: 1,
//                 b64_count: 8,
//                 ..Default::default()
//             },
//         }];

//         const ALIGN: usize = std::mem::align_of::<u64>();
//         const N: usize = 10;

//         let mut ctx = Context::new_raw(prog, desc);
//         let a = ctx.alloc(8 * N, ALIGN);
//         let b = ctx.alloc(8 * N, ALIGN);
//         let c = ctx.alloc(8 * N, ALIGN);

//         let data_a = vec![1u64; N];
//         let data_b = vec![2u64; N];
//         ctx.write(a, 0, bytemuck::cast_slice(&data_a));
//         ctx.write(b, 0, bytemuck::cast_slice(&data_b));

//         ctx.run(
//             LaunchParams::func_id(0).grid1d(1).block1d(N as u32),
//             &[Argument::Ptr(a), Argument::Ptr(b), Argument::Ptr(c)],
//         )
//         .unwrap();

//         let mut res = vec![0u64; N];
//         ctx.read(c, 0, bytemuck::cast_slice_mut(&mut res));

//         res.iter().for_each(|v| assert_eq!(*v, 3));
//     }
// }
