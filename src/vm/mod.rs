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
pub enum RegType {
    U64,
}

#[derive(Clone, Copy, Debug)]
pub struct RegOperand {
    ty: RegType,
    id: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum AddrOperand {
    Absolute(usize),
    AbsoluteReg(usize, RegOperand),
    StackRelative(isize),
    StackRelativeReg(isize, RegOperand),
    RegisterRelative(RegOperand, isize),
}

#[derive(Clone, Copy, Debug)]
pub enum Instruction {
    Load(MemStateSpace, RegOperand, AddrOperand),
    Store(MemStateSpace, RegOperand, AddrOperand),
    Move { ty: RegType, dst: RegOperand, src: RegOperand },
    Add { ty: RegType, dst: RegOperand, a: RegOperand, b: RegOperand },
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
struct RegDesc {
    b8_count: usize,
    b16_count: usize,
    b32_count: usize,
    b64_count: usize,
    b128_count: usize,
}

#[derive(Clone, Copy, Debug)]
struct FuncFrameDesc {
    iptr: IPtr,
    frame_size: usize,
    arg_size: usize,
    regs: RegDesc,
}

#[derive(Debug)]
pub struct Context {
    program: Vec<Instruction>,
    descriptors: Vec<FuncFrameDesc>,
}

#[derive(Debug)]
struct Registers {
    b8: Vec<[u8; 1]>,
    b16: Vec<[u8; 2]>,
    b32: Vec<[u8; 4]>,
    b64: Vec<[u8; 8]>,
    b128: Vec<[u8; 16]>,
}

impl Registers {
    fn new(desc: RegDesc) -> Registers {
        Registers {
            b8: vec![[0; 1]; desc.b8_count],
            b16: vec![[0; 2]; desc.b16_count],
            b32: vec![[0; 4]; desc.b32_count],
            b64: vec![[0; 8]; desc.b64_count],
            b128: vec![[0; 16]; desc.b128_count],
        }
    }

    fn get_u64(&self, id: usize) -> u64 {
        u64::from_ne_bytes(self.b64[id])
    }

    fn set_u64(&mut self, id: usize, value: u64) {
        self.b64[id] = value.to_ne_bytes();
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

    fn get_u64(&self, reg: RegOperand) -> u64 {
        self.regs.last().unwrap().get_u64(reg.id)
    }

    fn set_u64(&mut self, reg: RegOperand, value: u64) {
        self.regs.last_mut().unwrap().set_u64(reg.id, value);
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
                let offset = self.get_u64(reg) as isize;
                (addr as isize + offset) as usize
            }
            AddrOperand::StackRelative(offset) => {
                let frame_size = self.frame_meta.last().unwrap().frame_size as isize;
                (self.stack_data.len() as isize - frame_size + offset) as usize
            }
            AddrOperand::StackRelativeReg(offset, reg) => {
                let frame_size = self.frame_meta.last().unwrap().frame_size as isize;
                let reg_offset = self.get_u64(reg) as isize;
                (self.stack_data.len() as isize - frame_size + offset + reg_offset) as usize
            }
            AddrOperand::RegisterRelative(reg, offset) => {
                let reg_base = self.get_u64(reg) as isize;
                (reg_base + offset) as usize
            }
        }
    }
}

fn load_u64(data: &[u8], addr: usize) -> u64 {
    u64::from_ne_bytes(data[addr..addr + 8].try_into().unwrap())
}

fn store_u64(data: &mut [u8], addr: usize, value: u64) {
    data[addr..addr + 8].copy_from_slice(&value.to_ne_bytes());
}

impl Context {
    fn fetch_instr(&self, iptr: IPtr) -> Instruction {
        self.program[iptr.0]
    }

    pub fn run(&self, desc_id: usize, global_mem: &mut [u8], param_data: &[u8]) -> Result<(), ()> {
        let mut state = ThreadState::new();

        let desc = self.descriptors[desc_id];
        if param_data.len() != desc.arg_size {
            return Err(())
        }
        state.stack_data.resize(desc.arg_size, 0);
        state.stack_data.copy_from_slice(param_data);
        state.frame_setup(desc);

        while state.num_frames() > 0 {
            match self.fetch_instr(state.iptr_fetch_incr()) {
                Instruction::Load(space, dst, addr) => {
                    let addr = state.resolve_address(addr);
                    let value = match space {
                        MemStateSpace::Global => load_u64(global_mem, addr),
                        MemStateSpace::Stack => load_u64(&state.stack_data, addr),
                        MemStateSpace::Shared => {
                            todo!()
                        }
                    };
                    state.set_u64(dst, value);
                }
                Instruction::Store(space, src, addr) => {
                    let addr = state.resolve_address(addr);
                    let value = state.get_u64(src);
                    match space {
                        MemStateSpace::Global => {
                            store_u64(global_mem, addr, value);
                        }
                        MemStateSpace::Stack => {
                            store_u64(&mut state.stack_data, addr, value);
                        }
                        MemStateSpace::Shared => {
                            todo!();
                        }
                    }
                }
                Instruction::Add { ty: _, dst, a: src1, b: src2 } => {
                    let value1 = state.get_u64(src1);
                    let value2 = state.get_u64(src2);
                    state.set_u64(dst, value1 + value2);
                }
                Instruction::Move { ty: _, dst, src } => {
                    let value = state.get_u64(src);
                    state.set_u64(dst, value);
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
        let ctx = Context {
            program: vec![
                Instruction::Load(
                    MemStateSpace::Global,
                    RegOperand { ty: RegType::U64, id: 0 },
                    AddrOperand::Absolute(0),
                ),
                Instruction::Load(
                    MemStateSpace::Global,
                    RegOperand { ty: RegType::U64, id: 1 },
                    AddrOperand::Absolute(8),
                ),
                Instruction::Add{
                    ty: RegType::U64,
                    dst: RegOperand { ty: RegType::U64, id: 0 },
                    a: RegOperand { ty: RegType::U64, id: 0 },
                    b: RegOperand { ty: RegType::U64, id: 1 },
                },
                Instruction::Store(
                    MemStateSpace::Global,
                    RegOperand { ty: RegType::U64, id: 0 },
                    AddrOperand::Absolute(16),
                ),
                Instruction::Return,
            ],
            descriptors: vec![FuncFrameDesc {
                iptr: IPtr(0),
                frame_size: 0,
                arg_size: 0,
                regs: RegDesc {
                    b64_count: 2,
                    ..Default::default()
                }
            }],
        };
        let mut mem = vec![0u8; 24];
        store_u64(&mut mem, 0, 1);
        store_u64(&mut mem, 8, 2);
        ctx.run(0, &mut mem, &[]).unwrap();
        assert_eq!(load_u64(&mem, 16), 3);
    }

    #[test]
    fn simple_with_args() {
        let ctx = Context {
            program: vec![
                Instruction::Load(
                    MemStateSpace::Stack,
                    RegOperand { ty: RegType::U64, id: 0 },
                    AddrOperand::StackRelative(-16),
                ),
                Instruction::Load(
                    MemStateSpace::Stack,
                    RegOperand { ty: RegType::U64, id: 1 },
                    AddrOperand::StackRelative(-8),
                ),
                Instruction::Add{
                    ty: RegType::U64,
                    dst: RegOperand { ty: RegType::U64, id: 0 },
                    a: RegOperand { ty: RegType::U64, id: 0 },
                    b: RegOperand { ty: RegType::U64, id: 1 },
                },
                Instruction::Store(
                    MemStateSpace::Global,
                    RegOperand { ty: RegType::U64, id: 0 },
                    AddrOperand::Absolute(0),
                ),
                Instruction::Return,
            ],
            descriptors: vec![FuncFrameDesc {
                iptr: IPtr(0),
                frame_size: 0,
                arg_size: 16,
                regs: RegDesc {
                    b64_count: 2,
                    ..Default::default()
                }
            }],
        };
        let mut mem = vec![0u8; 8];
        let mut param_mem = vec![0u8; 16];
        store_u64(&mut param_mem, 0, 1);
        store_u64(&mut param_mem, 8, 2);
        ctx.run(0, &mut mem, &param_mem).unwrap();
        assert_eq!(load_u64(&mem, 0), 3);
    }
}