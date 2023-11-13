use std::collections::HashMap;

use crate::ast;
use crate::vm;

#[derive(Debug)]
pub struct CompiledModule {
    pub instructions: Vec<vm::Instruction>,
    pub func_descriptors: Vec<vm::FuncFrameDesc>,
    // pub global_vars: Vec<usize>,
    pub symbol_map: HashMap<String, vm::Symbol>,
}

#[derive(thiserror::Error, Debug)]
pub enum CompilationError {
    #[error("undefined symbol: {0:?}")]
    UndefinedSymbol(String),
    #[error("undefined label: {0:?}")]
    UndefinedLabel(String),
    #[error("invalid state space")]
    InvalidStateSpace,
    #[error("invalid operand: {0:?}")]
    InvalidOperand(ast::Operand),
    #[error("invalid immediate type")]
    InvalidImmediateType,
    #[error("invalid register type: {0:?}")]
    InvalidRegisterType(vm::RegOperand),
}

#[derive(Clone, Copy, Debug)]
enum Variable {
    Register(vm::RegOperand),
    Memory(vm::AddrOperand),
}
struct VariableMap(HashMap<String, Variable>);

impl VariableMap {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn insert(&mut self, ident: String, var: Variable) {
        self.0.insert(ident, var);
    }

    pub fn get(&self, ident: &str) -> Option<&Variable> {
        self.0.get(ident)
    }

    pub fn get_pred(&self, ident: &str) -> Result<vm::RegPred, CompilationError> {
        match self.0.get(ident) {
            Some(Variable::Register(reg)) => match reg {
                vm::RegOperand::Pred(pred) => Ok(*pred),
                _ => Err(CompilationError::InvalidRegisterType(*reg)),
            },
            Some(Variable::Memory(addr)) => Err(CompilationError::InvalidStateSpace),
            _ => Err(CompilationError::UndefinedSymbol(ident.to_string())),
        }
    }

    pub fn get_reg(&self, ident: &str) -> Result<vm::RegOperand, CompilationError> {
        match self.0.get(ident) {
            Some(Variable::Register(reg)) => Ok(*reg),
            Some(Variable::Memory(addr)) => Err(CompilationError::InvalidStateSpace),
            _ => Err(CompilationError::UndefinedSymbol(ident.to_string())),
        }
    }

    pub fn get_memory(&self, ident: &str) -> Result<vm::AddrOperand, CompilationError> {
        match self.0.get(ident) {
            Some(Variable::Register(reg)) => Err(CompilationError::InvalidStateSpace),
            Some(Variable::Memory(addr)) => Ok(*addr),
            _ => Err(CompilationError::UndefinedSymbol(ident.to_string())),
        }
    }
}

impl CompiledModule {
    fn compile_directive_toplevel(
        &mut self,
        directive: ast::Directive,
    ) -> Result<(), CompilationError> {
        use ast::Directive;
        match directive {
            Directive::VarDecl(_) => todo!(),
            Directive::Version(_) => Ok(()),
            Directive::Target(_) => Ok(()),
            Directive::AddressSize(a) => match a {
                ast::AddressSize::Adr64 => Ok(()),
                _ => todo!(),
            },
            Directive::Function(f) => self.compile_function(f),
        }
    }

    fn compile_function(&mut self, func: ast::Function) -> Result<(), CompilationError> {
        let iptr = vm::IPtr(self.instructions.len());
        let mut state = FuncCodegenState::new(self);
        state.compile_ast(func)?;
        let (ident, mut frame_desc, instructions) = state.finalize()?;
        frame_desc.iptr = iptr;
        self.instructions.extend(instructions);
        let desc = vm::Symbol::Function(self.func_descriptors.len());
        self.func_descriptors.push(frame_desc);
        self.symbol_map.insert(ident, desc);
        Ok(())
    }
}

struct FuncCodegenState<'a> {
    parent: &'a CompiledModule,
    ident: String,
    instructions: Vec<vm::Instruction>,
    var_map: VariableMap,
    regs: vm::RegDesc,
    stack_size: usize,
    shared_size: usize,
    param_stack_offset: usize,
    label_map: HashMap<String, usize>,
    jump_map: Vec<String>,
}

impl<'a> FuncCodegenState<'a> {
    pub fn new(parent: &'a CompiledModule) -> Self {
        Self {
            parent,
            ident: String::new(),
            instructions: Vec::new(),
            var_map: VariableMap::new(),
            regs: vm::RegDesc::default(),
            stack_size: 0,
            param_stack_offset: 0,
            shared_size: 0,
            label_map: HashMap::new(),
            jump_map: Vec::new(),
        }
    }

    fn declare_var(&mut self, decl: ast::VarDecl) -> Result<(), CompilationError> {
        use ast::StateSpace;
        use vm::RegOperand;
        if let ast::Type::Pred = decl.ty {
            // predicates can only exist in the reg state space
            if decl.state_space != StateSpace::Register {
                todo!()
            }
        }
        match decl.state_space {
            StateSpace::Register => {
                if !decl.array_bounds.is_empty() {
                    todo!("array bounds not supported on register variables")
                }
                use ast::Type::*;
                let opref = match decl.ty {
                    B128 => RegOperand::B128(self.regs.alloc_b128()),
                    B64 | U64 | S64 | F64 => RegOperand::B64(self.regs.alloc_b64()),
                    B32 | U32 | S32 | F32 | F16x2 => RegOperand::B32(self.regs.alloc_b32()),
                    B16 | U16 | S16 | F16 => RegOperand::B16(self.regs.alloc_b16()),
                    B8 | U8 | S8 => RegOperand::B8(self.regs.alloc_b8()),
                    Pred => RegOperand::Pred(self.regs.alloc_pred()),
                };
                self.var_map.insert(decl.ident, Variable::Register(opref));
            }
            StateSpace::Shared => {
                let count = decl.array_bounds.iter().product::<u32>();
                let size = decl.ty.size() * count as usize;
                let align = decl.ty.alignment();
                assert!(align.count_ones() == 1);
                // align to required alignment
                self.shared_size = (self.shared_size + align - 1) & !(align - 1);
                let loc = vm::AddrOperand::Absolute(self.shared_size);
                self.shared_size += size;
                self.var_map
                    .insert(decl.ident, Variable::Memory(loc));
            },
            StateSpace::Global => todo!(),
            StateSpace::Local => todo!(),
            StateSpace::Constant => todo!(),
            StateSpace::Parameter => todo!(),
        }
        Ok(())
    }

    fn handle_vars(&mut self, vars: Vec<ast::VarDecl>) -> Result<(), CompilationError> {
        for decl in vars {
            if let Some(mult) = decl.multiplicity {
                if !decl.array_bounds.is_empty() {
                    todo!("array bounds not supported on parametrized variables")
                }
                for i in 0..mult {
                    let mut decl = decl.clone();
                    decl.ident.push_str(&i.to_string());
                    decl.multiplicity = None;
                    self.declare_var(decl)?;
                }
            } else {
                self.declare_var(decl)?;
            }
        }
        Ok(())
    }

    fn handle_params(&mut self, params: Vec<ast::FunctionParam>) -> Result<(), CompilationError> {
        for param in params.iter().rev() {
            if matches!(param.ty, ast::Type::Pred) {
                // this should raise an error as predicates can only exist in the reg state space
                todo!()
            }

            // account for the size of the parameter
            self.param_stack_offset += param.ty.size();

            // align to required alignment
            assert!(param.ty.alignment().count_ones() == 1);
            self.param_stack_offset =
                (self.param_stack_offset + param.ty.alignment() - 1) & !(param.ty.alignment() - 1);

            let loc = vm::AddrOperand::StackRelative(-(self.param_stack_offset as isize));

            self.var_map
                .insert(param.ident.clone(), Variable::Memory(loc));
        }
        Ok(())
    }

    fn construct_immediate(
        &mut self,
        ty: ast::Type,
        imm: ast::Immediate,
    ) -> Result<vm::RegOperand, CompilationError> {
        use ast::Type::*;
        use vm::RegOperand;
        let opref = match ty {
            U32 | B32 => {
                let reg = RegOperand::B32(self.regs.alloc_b32());
                let ast::Immediate::Integer(val) = imm else {
                    return Err(CompilationError::InvalidImmediateType);
                };
                self.instructions
                    .push(vm::Instruction::Const(reg, vm::Constant::U32(val as u32)));
                reg
            }
            S32 => {
                let reg = RegOperand::B32(self.regs.alloc_b32());
                let ast::Immediate::Integer(val) = imm else {
                    return Err(CompilationError::InvalidImmediateType);
                };
                self.instructions
                    .push(vm::Instruction::Const(reg, vm::Constant::S32(val as i32)));
                reg
            }
            U64 | B64 => {
                let reg = RegOperand::B64(self.regs.alloc_b64());
                let ast::Immediate::Integer(val) = imm else {
                    return Err(CompilationError::InvalidImmediateType);
                };
                self.instructions
                    .push(vm::Instruction::Const(reg, vm::Constant::U64(val as u64)));
                reg
            }
            S64 => {
                let reg = RegOperand::B64(self.regs.alloc_b64());
                let ast::Immediate::Integer(val) = imm else {
                    return Err(CompilationError::InvalidImmediateType);
                };
                self.instructions
                    .push(vm::Instruction::Const(reg, vm::Constant::S64(val as i64)));
                reg
            }
            _ => todo!(),
        };
        Ok(opref)
    }

    fn get_src_reg(
        &mut self,
        ty: ast::Type,
        op: &ast::Operand,
    ) -> Result<vm::RegOperand, CompilationError> {
        use ast::Operand;
        match op {
            Operand::Variable(ident) => self.var_map.get_reg(ident),
            Operand::Immediate(imm) => self.construct_immediate(ty, *imm),
            Operand::SpecialReg(special) => Ok(vm::RegOperand::Special(*special)),
            Operand::Address(_) => todo!(),
        }
    }

    fn get_dst_reg(
        &mut self,
        ty: ast::Type,
        op: &ast::Operand,
    ) -> Result<vm::RegOperand, CompilationError> {
        use ast::Operand;
        match op {
            Operand::Variable(ident) => self.var_map.get_reg(ident),
            _ => Err(CompilationError::InvalidOperand(op.clone())),
        }
    }

    fn reg_pred_2src(
        &mut self,
        ty: ast::Type,
        ops: &[ast::Operand],
    ) -> Result<(vm::RegPred, vm::RegOperand, vm::RegOperand), CompilationError> {
        let [ast::Operand::Variable(dst), lhs_op, rhs_op] = ops else { todo!() };
        let dst_reg = self.var_map.get_pred(dst)?;
        let lhs_reg = self.get_src_reg(ty, lhs_op)?;
        let rhs_reg = self.get_src_reg(ty, rhs_op)?;
        Ok((dst_reg, lhs_reg, rhs_reg))
    }

    fn reg_dst_1src(
        &mut self,
        ty: ast::Type,
        ops: &[ast::Operand],
    ) -> Result<[vm::RegOperand; 2], CompilationError> {
        let [dst, src ] = ops else { todo!() };
        let dst_reg = self.get_dst_reg(ty, dst)?;
        let src_reg = self.get_src_reg(ty, src)?;
        Ok([dst_reg, src_reg])
    }

    fn reg_dst_2src(
        &mut self,
        ty: ast::Type,
        ops: &[ast::Operand],
    ) -> Result<[vm::RegOperand; 3], CompilationError> {
        let [dst, lhs_op, rhs_op] = ops else { todo!() };
        let dst_reg = self.get_dst_reg(ty, dst)?;
        let lhs_reg = self.get_src_reg(ty, lhs_op)?;
        let rhs_reg = self.get_src_reg(ty, rhs_op)?;
        Ok([dst_reg, lhs_reg, rhs_reg])
    }

    fn reg_dst_3src(
        &mut self,
        ty: ast::Type,
        ops: &[ast::Operand],
    ) -> Result<[vm::RegOperand; 4], CompilationError> {
        let [dst, src1_op, src2_op, src3_op] = ops else {
            todo!()
        };
        let dst_reg = self.get_dst_reg(ty, dst)?;
        let src1_reg = self.get_src_reg(ty, src1_op)?;
        let src2_reg = self.get_src_reg(ty, src2_op)?;
        let src3_reg = self.get_src_reg(ty, src3_op)?;
        Ok([dst_reg, src1_reg, src2_reg, src3_reg])
    }

    fn handle_instruction(&mut self, instr: ast::Instruction) -> Result<(), CompilationError> {
        use ast::AddressOperand;
        use ast::InstructionSpecifier::*;
        use ast::Operand;
        use vm::AddrOperand;

        if let Some(label) = instr.label {
            self.label_map.insert(label, self.instructions.len());
        }

        if let Some(guard) = instr.guard {
            let (ident, expected) = match guard {
                ast::Guard::Normal(s) => (s, false),
                ast::Guard::Negated(s) => (s, true),
            };
            let guard_reg = self.var_map.get_pred(&ident)?;
            self.instructions
                .push(vm::Instruction::SkipIf(guard_reg, expected));
        }

        match instr.specifier {
            Load(st, ty) => {
                let [Operand::Variable(ident), Operand::Address(addr_op)] =
                    instr.operands.as_slice()
                else {
                    todo!()
                };
                let dst_reg = self.var_map.get_reg(ident)?;
                let src_op = match self.var_map.get(addr_op.get_ident()) {
                    Some(Variable::Register(reg)) => {
                        match addr_op {
                            AddressOperand::Address(_) => {}
                            AddressOperand::AddressOffset(_, offset) => todo!(),
                            AddressOperand::AddressOffsetVar(_, ident) => todo!(),
                            AddressOperand::ArrayIndex(_, idx) => todo!(),
                        }
                        AddrOperand::RegisterRelative(*reg, 0)
                    }
                    Some(Variable::Memory(addr)) => {
                        match addr_op {
                            AddressOperand::Address(_) => {}
                            AddressOperand::AddressOffset(_, offset) => todo!(),
                            AddressOperand::AddressOffsetVar(_, ident) => todo!(),
                            AddressOperand::ArrayIndex(_, idx) => todo!(),
                        }
                        addr.clone()
                    }
                    None => {
                        return Err(CompilationError::UndefinedSymbol(
                            addr_op.get_ident().to_string(),
                        ))
                    }
                };
                // let base_addr = var_map.get_memory(addr_op.get_ident())?;
                self.instructions.push(vm::Instruction::Load(
                    // todo: handle invalid state space and get rid of panic
                    st.to_vm()
                        .unwrap_or_else(|| panic!("invalid state space: {:?}", st)),
                    dst_reg,
                    src_op,
                ))
            }
            Store(st, ty) => {
                let [Operand::Address(addr_op), Operand::Variable(ident)] =
                    instr.operands.as_slice()
                else {
                    todo!()
                };
                let src_reg = self.var_map.get_reg(ident)?;
                let dst_op = match self.var_map.get(addr_op.get_ident()) {
                    Some(Variable::Register(reg)) => {
                        match addr_op {
                            AddressOperand::Address(_) => {}
                            AddressOperand::AddressOffset(_, offset) => todo!(),
                            AddressOperand::AddressOffsetVar(_, ident) => todo!(),
                            AddressOperand::ArrayIndex(_, idx) => todo!(),
                        }
                        AddrOperand::RegisterRelative(*reg, 0)
                    }
                    Some(Variable::Memory(addr)) => {
                        match addr_op {
                            AddressOperand::Address(_) => {}
                            AddressOperand::AddressOffset(_, offset) => todo!(),
                            AddressOperand::AddressOffsetVar(_, ident) => todo!(),
                            AddressOperand::ArrayIndex(_, idx) => todo!(),
                        }
                        addr.clone()
                    }
                    None => {
                        return Err(CompilationError::UndefinedSymbol(
                            addr_op.get_ident().to_string(),
                        ))
                    }
                };
                // let base_addr = var_map.get_memory(addr_op.get_ident())?;
                self.instructions.push(vm::Instruction::Store(
                    // todo: handle invalid state space and get rid of panic
                    st.to_vm()
                        .unwrap_or_else(|| panic!("invalid state space: {:?}", st)),
                    src_reg,
                    dst_op,
                ))
            }
            Move(ty) => {
                let [dst, src] = instr.operands.as_slice() else {
                    todo!()
                };
                let dst_reg = self.get_dst_reg(ty, dst)?;
                let src_reg = match src {
                    Operand::Variable(ident) => {
                        match self.var_map.get(ident) {
                            Some(Variable::Register(reg)) => *reg,
                            // this is an LEA operation, not just a normal mov 
                            Some(Variable::Memory(addr)) => {
                                self.instructions.push(vm::Instruction::LoadEffectiveAddress(ty, dst_reg, *addr));
                                return Ok(())
                            }
                            // undefined variable
                            None => {
                                return Err(CompilationError::UndefinedSymbol(
                                    ident.to_string(),
                                ))
                            }
                        }
                    },
                    Operand::Immediate(imm) => self.construct_immediate(ty, *imm)?,
                    Operand::SpecialReg(special) => vm::RegOperand::Special(*special),
                    Operand::Address(_) => todo!(),
                };
                self.instructions
                    .push(vm::Instruction::Move(ty, dst_reg, src_reg));
            }
            Add(ty) => {
                let [dst_reg, lhs_reg, rhs_reg] =
                    self.reg_dst_2src(ty, instr.operands.as_slice())?;
                self.instructions
                    .push(vm::Instruction::Add(ty, dst_reg, lhs_reg, rhs_reg));
            }
            Multiply(mode, ty) => {
                let [dst_reg, lhs_reg, rhs_reg] =
                    self.reg_dst_2src(ty, instr.operands.as_slice())?;
                self.instructions
                    .push(vm::Instruction::Mul(ty, mode, dst_reg, lhs_reg, rhs_reg));
            }
            MultiplyAdd(mode, ty) => {
                let [dst, a, b, c] = self.reg_dst_3src(ty, &instr.operands)?;
                self.instructions.push(vm::Instruction::Mul(ty, mode, dst, a, b));
                self.instructions.push(vm::Instruction::Add(ty, dst, dst, c));
            }
            Convert { from, to } => {
                let [dst, src] = self.reg_dst_1src(from, &instr.operands)?;
                self.instructions.push(vm::Instruction::Convert {
                    dst_type: to,
                    src_type: from,
                    dst,
                    src,
                });
            },
            ConvertAddress(ty, st) => todo!(),
            ConvertAddressTo(ty, st) => {
                // for now, just move the address register into the destination register
                let [dst, src] = instr.operands.as_slice() else {
                    todo!();
                };
                let dst_reg = self.get_dst_reg(ty, dst)?;
                let src_reg = self.get_src_reg(ty, src)?;
                self.instructions
                    .push(vm::Instruction::Move(ty, dst_reg, src_reg));
            }
            SetPredicate(pred, ty) => {
                let (dst, a, b) = self.reg_pred_2src(ty, instr.operands.as_slice())?;
                self.instructions
                    .push(vm::Instruction::SetPredicate(ty, pred, dst, a, b));
            },
            ShiftLeft(ty) => {
                let [dst_reg, lhs_reg, rhs_reg] =
                    self.reg_dst_2src(ty, instr.operands.as_slice())?;
                self.instructions
                    .push(vm::Instruction::ShiftLeft(ty, dst_reg, lhs_reg, rhs_reg));
            },
            Call {
                uniform,
                ident,
                ret_param,
                params,
            } => todo!(),
            BarrierSync => {
                match instr.operands.as_slice() {
                    [idx] => {
                        let src_reg = self.get_src_reg(ast::Type::U32, idx)?;
                        self.instructions.push(vm::Instruction::BarrierSync {
                            idx: src_reg,
                            cnt: None,
                        })
                    },
                    [idx, cnt] => {
                        todo!()
                    }
                    _ => todo!()
                }
            }
            Branch => {
                let [Operand::Variable(ident)] = instr.operands.as_slice() else {
                    todo!()
                };
                let jump_idx = self.jump_map.len();
                self.jump_map.push(ident.clone());
                self.instructions.push(vm::Instruction::Jump {
                    offset: jump_idx as isize,
                });
            }
            Return => self.instructions.push(vm::Instruction::Return),
        };
        Ok(())
    }

    fn handle_instructions(
        &mut self,
        instructions: Vec<ast::Instruction>,
    ) -> Result<(), CompilationError> {
        for instr in instructions {
            self.handle_instruction(instr)?;
        }
        Ok(())
    }

    pub fn compile_ast(&mut self, func: ast::Function) -> Result<(), CompilationError> {
        self.ident = func.ident;
        let ast::Statement::Grouping(body) = *func.body else {
            todo!()
        };

        let mut instructions = Vec::new();
        let mut var_decls = Vec::new();
        for statement in body {
            use ast::{Directive, Statement};

            match statement {
                Statement::Directive(Directive::VarDecl(v)) => var_decls.push(v),
                Statement::Instruction(i) => instructions.push(i),
                Statement::Grouping(_) => todo!(),
                Statement::Directive(_) => todo!(),
            }
        }

        self.handle_params(func.params)?;
        self.handle_vars(var_decls)?;
        self.handle_instructions(instructions)?;

        Ok(())
    }

    pub fn finalize(
        mut self,
    ) -> Result<(String, vm::FuncFrameDesc, Vec<vm::Instruction>), CompilationError> {
        // resolve jump targets
        for (idx, instr) in self.instructions.iter_mut().enumerate() {
            if let vm::Instruction::Jump { offset } = instr {
                let jump_map_idx = *offset as usize;
                let target_label = &self.jump_map[jump_map_idx];
                if let Some(target_label) = self.label_map.get(target_label) {
                    *offset = *target_label as isize - idx as isize;
                } else {
                    return Err(CompilationError::UndefinedLabel(target_label.clone()));
                }
            }
        }

        // align stack size to 16 bytes
        self.stack_size = (self.stack_size + 15) & !15;
        let frame_desc = vm::FuncFrameDesc {
            iptr: vm::IPtr(self.instructions.len()),
            frame_size: self.stack_size,
            shared_size: 0,
            arg_size: self.param_stack_offset,
            regs: self.regs,
        };
        Ok((self.ident, frame_desc, self.instructions))
    }
}

pub fn compile(module: ast::Module) -> Result<CompiledModule, CompilationError> {
    let mut cmod = CompiledModule {
        instructions: Vec::new(),
        func_descriptors: Vec::new(),
        // global_vars: Vec::new(),
        symbol_map: HashMap::new(),
    };
    for directive in module.0 {
        cmod.compile_directive_toplevel(directive)?;
    }
    Ok(cmod)
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn compile_add_simple() {
        let contents = std::fs::read_to_string("kernels/add_simple.ptx").unwrap();
        let module = crate::ast::parse_program(&contents).unwrap();
        let _ = compile(module).unwrap();
    }

    #[test]
    fn compile_add() {
        let contents = std::fs::read_to_string("kernels/add.ptx").unwrap();
        let module = crate::ast::parse_program(&contents).unwrap();
        let _ = compile(module).unwrap();
    }

    #[test]
    fn compile_transpose() {
        let contents = std::fs::read_to_string("kernels/transpose.ptx").unwrap();
        let module = crate::ast::parse_program(&contents).unwrap();
        let _ = compile(module).unwrap();
    }
}