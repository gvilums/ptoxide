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
pub enum RegOperand {
    Pred(RegPred),
    B8(Reg8),
    B16(Reg16),
    B32(Reg32),
    B64(Reg64),
    B128(Reg128),
}

// #[derive(Clone, Copy, Debug)]
// pub enum RegOperand2<const N: usize> {
//     Pred([RegPred; N]),
//     B8([Reg8; N]),
//     B16([Reg16; N]),
//     B32([Reg32; N]),
//     B64([Reg64; N]),
//     B128([Reg128; N]),
// }

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
        a: RegOperand,
        b: RegOperand,
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

    pub fn write_global(&mut self, addr: usize, data: &[u8]) {
        if addr + data.len() > self.global_mem.len() {
            self.global_mem.resize(addr + data.len(), 0);
        }
        self.global_mem[addr..addr + data.len()].copy_from_slice(data);
    }

    pub fn read_global(&self, addr: usize, data: &mut [u8]) {
        data.copy_from_slice(&self.global_mem[addr..addr + data.len()]);
    }

    pub fn read_global_val<const N: usize>(&self, addr: usize) -> [u8; N] {
        self.global_mem[addr..addr + N].try_into().unwrap()
    }

    pub fn run(&mut self, desc_id: usize, param_data: &[u8]) -> Result<(), VmError> {
        let mut state = ThreadState::new();

        let desc = self.descriptors[desc_id];
        if param_data.len() != desc.arg_size {
            return Err(VmError::ParamDataSizeMismatch);
        }
        state.stack_data.resize(desc.arg_size, 0);
        state.stack_data.copy_from_slice(param_data);
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
                    }
                }
                Instruction::Add { ty: _, dst, a, b } => {
                    use RegOperand::*;
                    match (dst, a, b) {
                        (B64(dst), B64(a), B64(b)) => {
                            let a = u64::from_ne_bytes(state.get_b64(a));
                            let b = u64::from_ne_bytes(state.get_b64(b));
                            state.set_b64(dst, (a + b).to_ne_bytes());
                        }
                        _ => todo!(),
                    }
                }
                Instruction::Move { ty: _, dst, src } => {
                    use RegOperand::*;
                    match (dst, src) {
                        (B64(dst), B64(src)) => {
                            state.set_b64(dst, state.get_b64(src));
                        }
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
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn simple() {
        let prog = vec![
            Instruction::Load(
                MemStateSpace::Global,
                RegOperand::B64(Reg64 { id: 0 }),
                AddrOperand::Absolute(0),
            ),
            Instruction::Load(
                MemStateSpace::Global,
                RegOperand::B64(Reg64 { id: 1 }),
                AddrOperand::Absolute(8),
            ),
            Instruction::Add {
                ty: Type::U64,
                dst: RegOperand::B64(Reg64 { id: 0 }),
                a: RegOperand::B64(Reg64 { id: 0 }),
                b: RegOperand::B64(Reg64 { id: 1 }),
            },
            Instruction::Store(
                MemStateSpace::Global,
                RegOperand::B64(Reg64 { id: 0 }),
                AddrOperand::Absolute(16),
            ),
            Instruction::Return,
        ];
        let desc = vec![FuncFrameDesc {
            iptr: IPtr(0),
            frame_size: 0,
            arg_size: 0,
            regs: RegDesc {
                b64_count: 2,
                ..Default::default()
            },
        }];
        let mut ctx = Context::new(prog, desc);
        ctx.write_global(0, &1u64.to_ne_bytes());
        ctx.write_global(8, &2u64.to_ne_bytes());
        ctx.write_global(16, &0u64.to_ne_bytes());
        ctx.run(0, &[]).unwrap();
        assert_eq!(ctx.read_global_val(16), 3u64.to_ne_bytes());
    }

    // #[test]
    // fn simple_with_args() {
    //     let ctx = Context {
    //         program: vec![
    //             Instruction::Load(
    //                 MemStateSpace::Stack,
    //                 RegOperand::B64(Reg64 { id: 0 }),
    //                 AddrOperand::StackRelative(-16),
    //             ),
    //             Instruction::Load(
    //                 MemStateSpace::Stack,
    //                 RegOperand::B64(Reg64 { id: 1 }),
    //                 AddrOperand::StackRelative(-8),
    //             ),
    //             Instruction::Add {
    //                 ty: Type::U64,
    //                 dst: RegOperand::B64(Reg64 { id: 0 }),
    //                 a: RegOperand::B64(Reg64 { id: 0 }),
    //                 b: RegOperand::B64(Reg64 { id: 1 }),
    //             },
    //             Instruction::Store(
    //                 MemStateSpace::Global,
    //                 RegOperand::B64(Reg64 { id: 0 }),
    //                 AddrOperand::Absolute(0),
    //             ),
    //             Instruction::Return,
    //         ],
    //         descriptors: vec![FuncFrameDesc {
    //             iptr: IPtr(0),
    //             frame_size: 0,
    //             arg_size: 16,
    //             regs: RegDesc {
    //                 b64_count: 2,
    //                 ..Default::default()
    //             },
    //         }],
    //     };
    //     let mut mem = vec![0u8; 8];
    //     let mut param_mem = vec![0u8; 16];
    //     store_u64(&mut param_mem, 0, 1);
    //     store_u64(&mut param_mem, 8, 2);
    //     ctx.run(0, &mut mem, &param_mem).unwrap();
    //     assert_eq!(load_u64(&mem, 0), 3);
    // }
}
