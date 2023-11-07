pub mod bc;

#[derive(Clone, Copy, Debug)]
pub enum MemStateSpace {
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

#[derive(Clone, Copy, Debug)]
pub enum Type {
    Pred,
    U64,
    S64,
}

#[derive(Clone, Copy, Debug)]
pub enum PredicateOp {
    LessThan,
    LessThanEqual,
    Equal,
    NotEqual,
}

#[derive(Clone, Copy, Debug)]
pub enum SpecialReg {
    CtaIdX,
    NTIdX,
    TIdX,
}

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
pub enum Operand {
    Reg(RegOperand),
    Const(u64),
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
    Load(MemStateSpace, RegOperand, AddrOperand),
    Store(MemStateSpace, RegOperand, AddrOperand),
    Move {
        ty: Type,
        dst: RegOperand,
        src: RegOperand,
    },
    Add {
        ty: Type,
        dst: RegOperand,
        a: Operand,
        b: Operand,
    },
    Mul {
        ty: Type,
        dst: RegOperand,
        a: Operand,
        b: Operand,
    },
    SetPredicate {
        op: PredicateOp,
        dst: RegOperand,
        a: RegOperand,
        b: RegOperand,
    },
    Jump {
        offset: isize,
    },
    JumpIf {
        cond: RegPred,
        offset: isize,
    },
    Return,
}

#[derive(Clone, Copy, Debug)]
struct IPtr(usize);

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

#[derive(Clone, Copy, Debug)]
pub struct FuncFrameDesc {
    iptr: IPtr,
    frame_size: usize,
    arg_size: usize,
    regs: RegDesc,
}

#[derive(Debug)]
pub struct Context {
    global_mem: Vec<u8>,
    program: Vec<Instruction>,
    descriptors: Vec<FuncFrameDesc>,
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

impl ThreadState {
    fn new() -> ThreadState {
        ThreadState {
            iptr: IPtr(0),
            regs: Vec::new(),
            stack_data: Vec::new(),
            frame_meta: Vec::new(),
        }
    }

    fn get_pred(&self, reg: RegPred) -> bool {
        self.regs.last().unwrap().pred[reg.id]
    }

    fn set_pred(&mut self, reg: RegPred, value: bool) {
        self.regs.last_mut().unwrap().pred[reg.id] = value;
    }

    fn get_b64(&self, reg: Reg64) -> [u8; 8] {
        self.regs.last().unwrap().b64[reg.id]
    }

    fn set_b64(&mut self, reg: Reg64, value: [u8; 8]) {
        self.regs.last_mut().unwrap().b64[reg.id] = value;
    }

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
    #[error("parameter data did not match descriptor")]
    ParamDataSizeMismatch,
    #[error("slice size mismatch")]
    SliceSizeMismatch(#[from] std::array::TryFromSliceError),
}

#[derive(Clone, Copy, Debug)]
pub struct DevicePointer(u64);

pub enum Argument {
    Ptr(DevicePointer),
    Value(Vec<u8>),
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
        self.program[iptr.0]
    }

    pub fn new(program: Vec<Instruction>, descriptors: Vec<FuncFrameDesc>) -> Context {
        Context {
            global_mem: Vec::new(),
            program,
            descriptors,
        }
    }

    pub fn alloc(&mut self, size: usize) -> DevicePointer {
        let ptr = self.global_mem.len();
        self.global_mem.resize(ptr + size, 0);
        DevicePointer(ptr as u64)
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
            match self.fetch_instr(state.iptr_fetch_incr()) {
                Instruction::Load(space, dst, addr) => {
                    let addr = state.resolve_address(addr);
                    let data = match space {
                        MemStateSpace::Global => self.global_mem.as_slice(),
                        MemStateSpace::Stack => state.stack_data.as_slice(),
                        MemStateSpace::Shared => todo!(),
                    };
                    match dst {
                        RegOperand::Pred(r) => todo!(),
                        RegOperand::B8(r) => todo!(),
                        RegOperand::B16(r) => todo!(),
                        RegOperand::B32(r) => todo!(),
                        RegOperand::B64(r) => state.set_b64(r, data[addr..addr + 8].try_into()?),
                        RegOperand::B128(r) => todo!(),
                        RegOperand::Special(_) => todo!(),
                    }
                }
                Instruction::Store(space, src, addr) => {
                    let addr = state.resolve_address(addr);
                    // let value = state.get_u64(src);
                    match src {
                        RegOperand::Pred(r) => todo!(),
                        RegOperand::B8(r) => todo!(),
                        RegOperand::B16(r) => todo!(),
                        RegOperand::B32(r) => todo!(),
                        RegOperand::B64(r) => {
                            let val = state.get_b64(r);
                            let data = match space {
                                MemStateSpace::Global => self.global_mem.as_mut_slice(),
                                MemStateSpace::Stack => state.stack_data.as_mut_slice(),
                                MemStateSpace::Shared => todo!(),
                            };
                            data[addr..addr + 8].copy_from_slice(&val);
                        }
                        RegOperand::B128(r) => todo!(),
                        RegOperand::Special(_) => todo!(),
                    }
                }
                Instruction::Add { ty, dst, a, b } => {
                    use Operand::*;
                    use RegOperand::*;
                    match (dst, a, b) {
                        (B64(dst), Reg(B64(a)), Reg(B64(b))) => match ty {
                            Type::U64 => {
                                let a = u64::from_ne_bytes(state.get_b64(a));
                                let b = u64::from_ne_bytes(state.get_b64(b));
                                state.set_b64(dst, (a + b).to_ne_bytes());
                            }
                            _ => todo!(),
                        },
                        _ => todo!(),
                    }
                }
                Instruction::Mul { ty, dst, a, b } => {
                    use Operand::*;
                    use RegOperand::*;
                    match (dst, a, b) {
                        (B64(dst), Reg(B64(a)), Reg(B64(b))) => match ty {
                            Type::U64 => {
                                let a = u64::from_ne_bytes(state.get_b64(a));
                                let b = u64::from_ne_bytes(state.get_b64(b));
                                state.set_b64(dst, (a * b).to_ne_bytes());
                            }
                            _ => todo!(),
                        },
                        (B64(dst), Reg(B64(a)), Const(b)) => match ty {
                            Type::U64 => {
                                let a = u64::from_ne_bytes(state.get_b64(a));
                                state.set_b64(dst, (a * b).to_ne_bytes());
                            }
                            _ => todo!(),
                        },
                        _ => todo!(),
                    }
                }
                Instruction::Mul { ty, dst, a, b } => {
                    use Operand::*;
                    use RegOperand::*;
                    match (dst, a, b) {
                        (B64(dst), Reg(B64(a)), Reg(B64(b))) => match ty {
                            Type::U64 => {
                                let a = u64::from_ne_bytes(state.get_b64(a));
                                let b = u64::from_ne_bytes(state.get_b64(b));
                                state.set_b64(dst, (a + b).to_ne_bytes());
                            }
                            _ => todo!(),
                        },
                        _ => todo!(),
                    }
                }
                Instruction::Move { ty: _, dst, src } => {
                    use RegOperand::*;
                    match (dst, src) {
                        (B64(dst), B64(src)) => {
                            state.set_b64(dst, state.get_b64(src));
                        }
                        (B64(dst), Special(SpecialReg::TIdX)) => {
                            state.set_b64(dst, (tid.0 as u64).to_ne_bytes());
                        }
                        (B64(dst), Special(SpecialReg::NTIdX)) => {
                            state.set_b64(dst, (ntid.0 as u64).to_ne_bytes());
                        }
                        (B64(dst), Special(SpecialReg::CtaIdX)) => {
                            state.set_b64(dst, 0u64.to_ne_bytes());
                        }
                        // (B32(dst), Special(SpecialReg::TIdX)) => {
                        //     state.set_b32(dst, params.block_dim.0.to_ne_bytes());
                        // }
                        // (B32(dst), Special(SpecialReg::NTIdX)) => {
                        //     state.set_b32(dst, params.block_dim.0.to_ne_bytes());
                        // }
                        // (B32(dst), Special(SpecialReg::CtaIdX)) => {
                        //     state.set_b32(dst, params.block_dim.0.to_ne_bytes());
                        // }
                        _ => todo!(),
                    }
                }
                Instruction::SetPredicate { op, dst, a, b } => {
                    todo!()
                }
                Instruction::Jump { offset } => {
                    state.iptr.0 = (state.iptr.0 as isize + offset) as usize;
                }
                Instruction::JumpIf { cond, offset } => {
                    if state.get_pred(cond) {
                        state.iptr.0 = (state.iptr.0 as isize + offset) as usize;
                    }
                }
                Instruction::Return => state.frame_teardown(),
            }
        }
        Ok(())
    }

    pub fn run(&mut self, params: LaunchParams, args: &[Argument]) -> Result<(), VmError> {
        // for now we only support a single CTA
        assert!(params.grid_dim.0 == 1 && params.grid_dim.1 == 1 && params.grid_dim.2 == 1);

        let desc = self.descriptors[params.func_id];
        let arg_size: usize = args
            .iter()
            .map(|arg| match arg {
                Argument::Ptr(_) => std::mem::size_of::<u64>(),
                Argument::Value(v) => v.len(),
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
                Argument::Value(v) => {
                    init_stack.extend_from_slice(v);
                }
            }
        }
        for x in 0..params.block_dim.0 {
            for y in 0..params.block_dim.1 {
                for z in 0..params.block_dim.2 {
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
                MemStateSpace::Stack,
                RegOperand::B64(Reg64 { id: 0 }),
                AddrOperand::StackRelative(-24),
            ),
            Instruction::Load(
                MemStateSpace::Stack,
                RegOperand::B64(Reg64 { id: 1 }),
                AddrOperand::StackRelative(-16),
            ),
            Instruction::Load(
                MemStateSpace::Stack,
                RegOperand::B64(Reg64 { id: 2 }),
                AddrOperand::StackRelative(-8),
            ),
            // load values from memory (pointed to by arguments)
            Instruction::Load(
                MemStateSpace::Global,
                RegOperand::B64(Reg64 { id: 3 }),
                AddrOperand::RegisterRelative(Reg64 { id: 0 }, 0),
            ),
            Instruction::Load(
                MemStateSpace::Global,
                RegOperand::B64(Reg64 { id: 4 }),
                AddrOperand::RegisterRelative(Reg64 { id: 1 }, 0),
            ),
            // add values
            Instruction::Add {
                ty: Type::U64,
                dst: RegOperand::B64(Reg64 { id: 5 }),
                a: Operand::Reg(RegOperand::B64(Reg64 { id: 3 })),
                b: Operand::Reg(RegOperand::B64(Reg64 { id: 4 })),
            },
            // store result
            Instruction::Store(
                MemStateSpace::Global,
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
        let mut ctx = Context::new(prog, desc);
        let a = ctx.alloc(8);
        let b = ctx.alloc(8);
        let c = ctx.alloc(8);
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
                MemStateSpace::Stack,
                RegOperand::B64(Reg64 { id: 0 }),
                AddrOperand::StackRelative(-24),
            ),
            Instruction::Load(
                MemStateSpace::Stack,
                RegOperand::B64(Reg64 { id: 1 }),
                AddrOperand::StackRelative(-16),
            ),
            Instruction::Load(
                MemStateSpace::Stack,
                RegOperand::B64(Reg64 { id: 2 }),
                AddrOperand::StackRelative(-8),
            ),
            // load thread index
            Instruction::Move {
                ty: Type::U64,
                dst: RegOperand::B64(Reg64 { id: 6 }),
                src: RegOperand::Special(SpecialReg::TIdX),
            },
            // multiply thread index by 8 (size of u64)
            Instruction::Mul {
                ty: Type::U64,
                dst: RegOperand::B64(Reg64 { id: 6 }),
                a: Operand::Reg(RegOperand::B64(Reg64 { id: 6 })),
                b: Operand::Const(8),
            },
            // offset argument pointers by thread index
            Instruction::Add {
                ty: Type::U64,
                dst: RegOperand::B64(Reg64 { id: 0 }),
                a: Operand::Reg(RegOperand::B64(Reg64 { id: 0 })),
                b: Operand::Reg(RegOperand::B64(Reg64 { id: 6 })),
            },
            Instruction::Add {
                ty: Type::U64,
                dst: RegOperand::B64(Reg64 { id: 1 }),
                a: Operand::Reg(RegOperand::B64(Reg64 { id: 1 })),
                b: Operand::Reg(RegOperand::B64(Reg64 { id: 6 })),
            },
            Instruction::Add {
                ty: Type::U64,
                dst: RegOperand::B64(Reg64 { id: 2 }),
                a: Operand::Reg(RegOperand::B64(Reg64 { id: 2 })),
                b: Operand::Reg(RegOperand::B64(Reg64 { id: 6 })),
            },
            // load values from memory (pointed to by offset arguments)
            Instruction::Load(
                MemStateSpace::Global,
                RegOperand::B64(Reg64 { id: 3 }),
                AddrOperand::RegisterRelative(Reg64 { id: 0 }, 0),
            ),
            Instruction::Load(
                MemStateSpace::Global,
                RegOperand::B64(Reg64 { id: 4 }),
                AddrOperand::RegisterRelative(Reg64 { id: 1 }, 0),
            ),
            // add values
            Instruction::Add {
                ty: Type::U64,
                dst: RegOperand::B64(Reg64 { id: 5 }),
                a: Operand::Reg(RegOperand::B64(Reg64 { id: 3 })),
                b: Operand::Reg(RegOperand::B64(Reg64 { id: 4 })),
            },
            // store result
            Instruction::Store(
                MemStateSpace::Global,
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
                b64_count: 7,
                ..Default::default()
            },
        }];
        const N: usize = 10;
        let mut ctx = Context::new(prog, desc);
        let a = ctx.alloc(8 * N);
        let b = ctx.alloc(8 * N);
        let c = ctx.alloc(8 * N);
        for i in 0..N {
            ctx.write(a, 8 * i, &1u64.to_ne_bytes());
            ctx.write(b, 8 * i, &2u64.to_ne_bytes());
        }
        ctx.run(
            LaunchParams::new().func(0).grid1d(1).block1d(N as u32),
            &[Argument::Ptr(a), Argument::Ptr(b), Argument::Ptr(c)],
        )
        .unwrap();
        for i in 0..N {
            let mut res = [0u8; 8];
            ctx.read(c, 8 * i, &mut res);
            assert_eq!(u64::from_ne_bytes(res), 3);
        }
    }
}
