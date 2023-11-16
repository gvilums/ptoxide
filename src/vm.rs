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

use crate::ast::PredicateOp;
use crate::ast::Type;

#[derive(Clone, Copy, Debug)]
pub enum Constant {
    U64(u64),
    S64(i64),
    F64(f64),
    F32(f32),
}

#[derive(Clone, Copy, Debug)]
pub struct GenericReg(pub usize);

#[derive(Clone, Copy, Debug)]
pub enum RegOperand {
    Generic(GenericReg),
    Special(SpecialReg),
}

impl From<GenericReg> for RegOperand {
    fn from(value: GenericReg) -> Self {
        RegOperand::Generic(value)
    }
}

impl From<SpecialReg> for RegOperand {
    fn from(value: SpecialReg) -> Self {
        RegOperand::Special(value)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Instruction {
    Load(Type, StateSpace, GenericReg, RegOperand),
    Store(Type, StateSpace, RegOperand, RegOperand),
    Move(Type, GenericReg, RegOperand),
    Const(GenericReg, Constant),
    Convert {
        dst_type: Type,
        src_type: Type,
        dst: GenericReg,
        src: RegOperand,
    },

    Add(Type, GenericReg, RegOperand, RegOperand),
    Sub(Type, GenericReg, RegOperand, RegOperand),
    Or(Type, GenericReg, RegOperand, RegOperand),
    And(Type, GenericReg, RegOperand, RegOperand),
    ShiftLeft(Type, GenericReg, RegOperand, RegOperand),
    Mul(Type, MulMode, GenericReg, RegOperand, RegOperand),
    Neg(Type, GenericReg, RegOperand),

    PushArg(RegOperand),
    PopArg(Option<GenericReg>),
    Call(usize),

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
    SetPredicate(Type, PredicateOp, GenericReg, RegOperand, RegOperand),
    SkipIf(RegOperand, bool),

    Return,
}

#[derive(Clone, Copy, Debug)]
pub struct IPtr(pub usize);

#[derive(Clone, Debug)]
struct FrameMeta {
    return_addr: IPtr,
    num_args: usize,
    reg_base: usize,
    stack_base: usize,
}

#[derive(Debug, Clone)]
struct ThreadState {
    iptr: IPtr,
    regs: Vec<u128>,
    stack_data: Vec<u8>,
    stack_frames: Vec<FrameMeta>,
    nctaid: (u32, u32, u32),
    ctaid: (u32, u32, u32),
    ntid: (u32, u32, u32),
    tid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug)]
pub struct FuncFrameDesc {
    pub iptr: IPtr,
    pub frame_size: usize,
    pub num_args: usize,
    pub num_regs: usize,
    pub shared_size: usize,
}

#[derive(Debug, Default)]
pub struct Context {
    global_mem: Vec<u8>,
    instructions: Vec<Instruction>,
    descriptors: Vec<FuncFrameDesc>,
    symbol_map: HashMap<String, Symbol>,
}

macro_rules! byte_reg_funcs {
    ($($get:ident, $set:ident, $helper_fn:ident, $helper_type:ty, $n:expr);* $(;)?) => {
        $(
            fn $get(&self, reg: RegOperand) -> [u8; $n] {
                self.$helper_fn(reg).to_ne_bytes()
            }

            fn $set(&mut self, reg: GenericReg, value: [u8; $n]) {
                self.set(reg, <$helper_type>::from_ne_bytes(value) as u128);
            }
        )*
    };
}

macro_rules! int_getters {
    ($($get:ident, $t2:ty);* $(;)?) => {
        $(
            fn $get(&self, reg: RegOperand) -> $t2 {
                self.get(reg) as $t2
            }
        )*
    };
}

macro_rules! int_setters {
    ($($set:ident, $t2:ty);* $(;)?) => {
        $(
            fn $set(&mut self, reg: GenericReg, value: $t2) {
                self.set(reg, value as u128);
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

    fn get(&self, reg: RegOperand) -> u128 {
        match reg {
            RegOperand::Generic(reg) => {
                let meta = self.stack_frames.last().unwrap();
                let idx = meta.reg_base - meta.num_args + reg.0;
                self.regs[idx]
            }
            RegOperand::Special(reg) => self.get_special(reg),
        }
    }

    fn set(&mut self, reg: GenericReg, val: u128) {
        let meta = self.stack_frames.last().unwrap();
        let idx = meta.reg_base - meta.num_args + reg.0;
        self.regs[idx] = val;
    }

    fn get_pred(&self, reg: RegOperand) -> bool {
        self.get(reg) != 0
    }

    fn get_f32(&self, reg: RegOperand) -> f32 {
        f32::from_bits(self.get_u32(reg))
    }

    fn set_f32(&mut self, reg: GenericReg, val: f32) {
        self.set_u32(reg, val.to_bits())
    }

    fn get_f64(&self, reg: RegOperand) -> f64 {
        f64::from_bits(self.get_u64(reg))
    }

    fn set_f64(&mut self, reg: GenericReg, val: f64) {
        self.set_u64(reg, val.to_bits())
    }

    byte_reg_funcs!(
        // get_b8, set_b8, get_u8, u8, 1;
        // get_b16, set_b16, get_u16, u16, 2;
        // get_b32, set_b32, get_u32, u32, 4;
        // get_b64, set_b64, get_u64, u64, 8;
        get_b128, set_b128, get_u128, u128, 16;
    );

    int_getters!(
        // get_u8, u8;
        get_u16, u16;
        get_u32, u32;
        get_u64, u64;
        get_u128, u128;

        // get_i8, i8;
        // get_i16, i16;
        get_i32, i32;
        get_i64, i64;
        // get_i128, i128;
    );
    int_setters!(
        set_pred, bool;
        // set_u8, u8;
        set_u16, u16;
        set_u32, u32;
        set_u64, u64;
        // set_u128, u128;

        // set_i8, i8;
        // set_i16, i16;
        set_i32, i32;
        set_i64, i64;
        // set_i128, i128;
    );

    fn iptr_fetch_incr(&mut self) -> IPtr {
        let ret = self.iptr;
        self.iptr.0 += 1;
        ret
    }

    fn frame_teardown(&mut self) {
        let meta = self.stack_frames.pop().unwrap();
        self.stack_data
            .truncate(meta.stack_base);
        self.regs.resize(meta.reg_base + meta.num_args, 0);
        self.iptr = meta.return_addr;
    }

    fn frame_setup(&mut self, desc: FuncFrameDesc) {
        self.stack_frames.push(FrameMeta {
            return_addr: self.iptr,
            num_args: desc.num_args,
            reg_base: self.regs.len(),
            stack_base: self.stack_data.len(),
        });
        self.stack_data
            .resize(self.stack_data.len() + desc.frame_size, 0);
        // args were already set up by caller
        self.regs.resize(self.regs.len() + desc.num_regs, 0);
        self.iptr = desc.iptr;
    }

    fn read_reg_unsigned(&self, reg: RegOperand) -> VmResult<usize> {
        Ok(self.get(reg) as usize)
    }

    fn get_stack_ptr(&self) -> usize {
        self.stack_frames.last().unwrap().stack_base
    }

    fn get_special(&self, s: SpecialReg) -> u128 {
        match s {
            SpecialReg::StackPtr => self.get_stack_ptr() as u128,
            SpecialReg::ThreadId => todo!(),
            SpecialReg::ThreadIdX => self.tid.0 as u128,
            SpecialReg::ThreadIdY => self.tid.1 as u128,
            SpecialReg::ThreadIdZ => self.tid.2 as u128,
            SpecialReg::NumThread => todo!(),
            SpecialReg::NumThreadX => self.ntid.0 as u128,
            SpecialReg::NumThreadY => self.ntid.1 as u128,
            SpecialReg::NumThreadZ => self.ntid.2 as u128,
            SpecialReg::CtaId => todo!(),
            SpecialReg::CtaIdX => self.ctaid.0 as u128,
            SpecialReg::CtaIdY => self.ctaid.1 as u128,
            SpecialReg::CtaIdZ => self.ctaid.2 as u128,
            SpecialReg::NumCta => todo!(),
            SpecialReg::NumCtaX => self.nctaid.0 as u128,
            SpecialReg::NumCtaY => self.nctaid.1 as u128,
            SpecialReg::NumCtaZ => self.nctaid.2 as u128,
        }
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
    barriers: Vec<Option<Barrier>>,
}

impl Barriers {
    pub fn new() -> Self {
        Barriers {
            barriers: Vec::new(),
        }
    }

    pub fn arrive(&mut self, _idx: usize, _target: usize) -> VmResult<Vec<ThreadState>> {
        todo!()
    }

    pub fn block(
        &mut self,
        idx: usize,
        target: usize,
        thread: ThreadState,
    ) -> VmResult<Vec<ThreadState>> {
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
        $($target_ty:pat, $op:ident, $getter:ident, $setter:ident);*$(;)?) => {
        match ($tyop) {
        $(
            $target_ty => {
                let val = $threadop.$getter($aop).$op($threadop.$getter($bop));
                $threadop.$setter($dstop, val);
            }
        )*
            _ => todo!()
        }
    };
}

macro_rules! unary_op {
    ($threadop:expr, $tyop:expr, $dstop:expr, $srcop:expr;
        $($target_ty:pat, $op:ident, $getter:ident, $setter:ident);*$(;)?) => {
        match ($tyop) {
        $(
            $target_ty => {
                let val = $threadop.$getter($srcop).$op();
                $threadop.$setter($dstop, val);
            }
        )*
            _ => todo!()
        }
    };
}

macro_rules! comparison_op {
    ($threadop:expr, $tyop:expr, $op:expr, $dstop:expr, $aop:expr, $bop:expr;
        $($target_ty:pat, $getter:ident);*$(;)?) => {
        match ($tyop) {
        $(
            $target_ty => {
                let a = $threadop.$getter($aop);
                let b = $threadop.$getter($bop);
                let value = match $op {
                    PredicateOp::LessThan => a < b,
                    PredicateOp::LessThanEqual => a <= b,
                    PredicateOp::Equal => a == b,
                    PredicateOp::NotEqual => a != b,
                    PredicateOp::GreaterThan => a > b,
                    PredicateOp::GreaterThanEqual => a >= b,
                };
                $threadop.set_pred($dstop, value);
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

    pub fn reset_mem(&mut self) {
        self.global_mem.clear();
    }

    fn run_cta(
        &mut self,
        nctaid: (u32, u32, u32),
        ctaid: (u32, u32, u32),
        ntid: (u32, u32, u32),
        desc: FuncFrameDesc,
        init_stack: &[u8],
        init_regs: &[u128],
    ) -> VmResult<()> {
        let mut shared_mem = vec![0u8; desc.shared_size];

        let mut runnable = Vec::new();
        for x in 0..ntid.0 {
            for y in 0..ntid.1 {
                for z in 0..ntid.2 {
                    let mut state = ThreadState::new(nctaid, ctaid, ntid, (x, y, z));
                    state.stack_data.extend_from_slice(init_stack);
                    state.regs.extend_from_slice(init_regs);
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
                        continue;
                    }
                    ThreadResult::Sync(idx, cnt) => {
                        let cnt = cnt.unwrap_or(cta_size);
                        runnable.extend(barriers.block(idx, cnt, state)?);
                        break;
                    }
                    ThreadResult::Exit => break,
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
            Instruction::Load(ty, space, dst, addr) => {
                let addr = thread.read_reg_unsigned(addr)?;
                let mut buf = [0u8; 16];
                let data = match space {
                    StateSpace::Global => self.global_mem.as_slice(),
                    StateSpace::Stack => thread.stack_data.as_slice(),
                    StateSpace::Shared => shared_mem,
                };
                let size = ty.size();
                buf[..size].copy_from_slice(&data[addr..addr + size]);
                thread.set_b128(dst, buf);
            }
            Instruction::Store(ty, space, src, addr) => {
                let addr = thread.read_reg_unsigned(addr)?;
                let buf = thread.get_b128(src);
                let data = match space {
                    StateSpace::Global => self.global_mem.as_mut_slice(),
                    StateSpace::Stack => thread.stack_data.as_mut_slice(),
                    StateSpace::Shared => shared_mem,
                };
                let size = ty.size();
                data[addr..addr + size].copy_from_slice(&buf[..size]);
            }
            Instruction::Sub(ty, dst, a, b) => {
                use std::ops::Sub;
                binary_op! {
                    thread, ty, dst, a, b;
                    Type::U64 | Type::B64, sub, get_u64, set_u64;
                    Type::S64, sub, get_i64, set_i64;
                    Type::U32 | Type::B32, sub, get_u32, set_u32;
                    Type::S32, sub, get_i32, set_i32;
                    Type::F32, sub, get_f32, set_f32;
                };
            }
            Instruction::Add(ty, dst, a, b) => {
                use std::ops::Add;
                binary_op! {
                    thread, ty, dst, a, b;
                    Type::U64 | Type::B64, add, get_u64, set_u64;
                    Type::S64, add, get_i64, set_i64;
                    Type::U32 | Type::B32, add, get_u32, set_u32;
                    Type::S32, add, get_i32, set_i32;
                    Type::F64, add, get_f64, set_f64;
                    Type::F32, add, get_f32, set_f32;
                };
            }
            Instruction::Mul(ty, mode, dst, a, b) => {
                use std::ops::Mul;
                match mode {
                    MulMode::Low => {
                        binary_op! {
                            thread, ty, dst, a, b;
                            Type::U64 | Type::B64, mul, get_u64, set_u64;
                            Type::S64, mul, get_i64, set_i64;
                            Type::U32 | Type::B32, mul, get_u32, set_u32;
                            Type::S32, mul, get_i32, set_i32;
                            Type::F32, mul, get_f32, set_f32;
                        }
                    }
                    MulMode::High => todo!(),
                    MulMode::Wide => match ty {
                        Type::U32 => {
                            thread
                                .set_u64(dst, thread.get_u32(a) as u64 * thread.get_u32(b) as u64);
                        }
                        Type::S32 => {
                            thread
                                .set_i64(dst, thread.get_i32(a) as i64 * thread.get_i32(b) as i64);
                        }
                        _ => todo!(),
                    },
                }
            }
            Instruction::Or(ty, dst, a, b) => {
                use std::ops::BitOr;
                binary_op! {
                    thread, ty, dst, a, b;
                    Type::Pred, bitor, get_pred, set_pred;
                    Type::U64 | Type::B64 | Type::S64, bitor, get_u64, set_u64;
                    Type::U32 | Type::B32 | Type::S32, bitor, get_u32, set_u32;
                };
            }
            Instruction::And(ty, dst, a, b) => {
                use std::ops::BitAnd;
                binary_op! {
                    thread, ty, dst, a, b;
                    Type::Pred, bitand, get_pred, set_pred;
                    Type::U64 | Type::B64 | Type::S64, bitand, get_u64, set_u64;
                    Type::U32 | Type::B32 | Type::S32, bitand, get_u32, set_u32;
                };
            }
            Instruction::Neg(ty, dst, src) => {
                use std::ops::Neg;
                unary_op! {
                    thread, ty, dst, src;
                    Type::S64, neg, get_i64, set_i64;
                    Type::S32, neg, get_i32, set_i32;
                    Type::F32, neg, get_f32, set_f32;
                };
            }
            Instruction::ShiftLeft(ty, dst, a, b) => {
                use std::ops::Shl;
                binary_op! {
                    thread, ty, dst, a, b;
                    Type::B64, shl, get_u64, set_u64;
                    Type::B32, shl, get_u32, set_u32;
                    Type::B16, shl, get_u16, set_u16;
                };
            }
            Instruction::Convert {
                dst_type,
                src_type,
                dst,
                src,
            } => match (dst_type, src_type) {
                (Type::U64, Type::U32) => {
                    thread.set_u64(dst, thread.get_u32(src) as u64);
                }
                _ => todo!(),
            },
            Instruction::Move(_, dst, src) => {
                thread.set(dst, thread.get(src));
            }
            Instruction::Const(dst, value) => match value {
                Constant::U64(value) => thread.set_u64(dst, value),
                Constant::S64(value) => thread.set_i64(dst, value),
                Constant::F32(value) => thread.set_f32(dst, value),
                Constant::F64(value) => thread.set_f64(dst, value),
            },
            Instruction::SetPredicate(ty, op, dst, a, b) => {
                comparison_op! {
                    thread, ty, op, dst, a, b;
                    Type::U64 | Type::B64, get_u64;
                    Type::S64, get_i64;
                    Type::U32 | Type::B32, get_u32;
                    Type::S32, get_i32;
                    Type::F32, get_f32;
                };
            }
            Instruction::BarrierSync { idx, cnt } => {
                let idx = thread.get_u32(idx) as usize;
                let cnt = cnt.map(|r| thread.get_u64(r) as usize);
                return Ok(ThreadResult::Sync(idx, cnt));
            }
            Instruction::BarrierArrive { idx, cnt } => {
                let idx = thread.get_u32(idx) as usize;
                let cnt = cnt.map(|r| thread.get_u64(r) as usize);
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
            Instruction::PushArg(reg) => {
                let val = thread.get(reg);
                thread.regs.push(val);
            },
            Instruction::PopArg(reg) => {
                let val = thread.regs.pop().unwrap();
                if let Some(reg) = reg {
                    thread.set(reg, val);
                }
            },
            Instruction::Call(desc_idx) => {
                let desc = self.descriptors[desc_idx];
                thread.frame_setup(desc);
            },
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
            }
            FuncIdent::Id(i) => *self
                .descriptors
                .get(i)
                .ok_or(VmError::InvalidFunctionId(i))?,
        };
        if args.len() != desc.num_args {
            return Err(VmError::ParamDataSizeMismatch);
        }

        let mut init_stack = Vec::new();
        let mut init_regs = Vec::new();
        for arg in args {
            init_regs.push(init_stack.len() as u128);
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
                        &init_regs,
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
            Instruction::Load(
                Type::U64,
                StateSpace::Stack,
                GenericReg(0),
                GenericReg(0).into(),
            ),
            Instruction::Load(
                Type::U64,
                StateSpace::Stack,
                GenericReg(1),
                GenericReg(1).into(),
            ),
            Instruction::Load(
                Type::U64,
                StateSpace::Stack,
                GenericReg(2),
                GenericReg(2).into(),
            ),
            Instruction::Load(
                Type::U64,
                StateSpace::Global,
                GenericReg(0),
                GenericReg(0).into(),
            ),
            Instruction::Load(
                Type::U64,
                StateSpace::Global,
                GenericReg(1),
                GenericReg(1).into(),
            ),
            // add values
            Instruction::Add(
                Type::U64,
                GenericReg(0),
                GenericReg(0).into(),
                GenericReg(1).into(),
            ),
            // store result
            Instruction::Store(
                Type::U64,
                StateSpace::Global,
                GenericReg(0).into(),
                GenericReg(2).into(),
            ),
            Instruction::Return,
        ];
        let desc = vec![FuncFrameDesc {
            iptr: IPtr(0),
            frame_size: 0,
            shared_size: 0,
            num_args: 3,
            num_regs: 0,
        }];
        const ALIGN: usize = std::mem::align_of::<u64>();
        let mut ctx = Context::new_raw(prog, desc);
        let a = ctx.alloc(8, ALIGN);
        let b = ctx.alloc(8, ALIGN);
        let c = ctx.alloc(8, ALIGN);
        ctx.write(a, 0, &1u64.to_ne_bytes());
        ctx.write(b, 0, &2u64.to_ne_bytes());
        ctx.run(
            LaunchParams::func_id(0).grid1d(1).block1d(1),
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
            Instruction::Load(
                Type::U64,
                StateSpace::Stack,
                GenericReg(0),
                GenericReg(0).into(),
            ),
            Instruction::Load(
                Type::U64,
                StateSpace::Stack,
                GenericReg(1),
                GenericReg(1).into(),
            ),
            Instruction::Load(
                Type::U64,
                StateSpace::Stack,
                GenericReg(2),
                GenericReg(2).into(),
            ),
            Instruction::Const(
                GenericReg(3),
                Constant::U64(8),
            ),
            Instruction::Mul(
                Type::U64,
                MulMode::Low,
                GenericReg(3),
                GenericReg(3).into(),
                SpecialReg::ThreadIdX.into(),
            ),
            Instruction::Add(
                Type::U64,
                GenericReg(0),
                GenericReg(0).into(),
                GenericReg(3).into(),
            ),
            Instruction::Add(
                Type::U64,
                GenericReg(1),
                GenericReg(1).into(),
                GenericReg(3).into(),
            ),
            Instruction::Add(
                Type::U64,
                GenericReg(2),
                GenericReg(2).into(),
                GenericReg(3).into(),
            ),
            Instruction::Load(
                Type::U64,
                StateSpace::Global,
                GenericReg(0),
                GenericReg(0).into(),
            ),
            Instruction::Load(
                Type::U64,
                StateSpace::Global,
                GenericReg(1),
                GenericReg(1).into(),
            ),
            // add values
            Instruction::Add(
                Type::U64,
                GenericReg(0),
                GenericReg(0).into(),
                GenericReg(1).into(),
            ),
            // store result
            Instruction::Store(
                Type::U64,
                StateSpace::Global,
                GenericReg(0).into(),
                GenericReg(2).into(),
            ),
            Instruction::Return,
        ];
        let desc = vec![FuncFrameDesc {
            iptr: IPtr(0),
            frame_size: 0,
            shared_size: 0,
            num_args: 3,
            num_regs: 1,
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
            LaunchParams::func_id(0).grid1d(1).block1d(N as u32),
            &[Argument::Ptr(a), Argument::Ptr(b), Argument::Ptr(c)],
        )
        .unwrap();

        let mut res = vec![0u64; N];
        ctx.read(c, 0, bytemuck::cast_slice_mut(&mut res));

        res.iter().for_each(|v| assert_eq!(*v, 3));
    }
}
