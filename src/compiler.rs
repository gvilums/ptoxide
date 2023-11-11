use std::collections::HashMap;

use crate::ast;
use crate::vm;

pub enum Symbol {
    Function(usize),
    Variable(usize),
}

pub struct CompiledModule {
    pub instructions: Vec<vm::Instruction>,
    pub func_descriptors: Vec<vm::FuncFrameDesc>,
    pub global_vars: Vec<usize>,
    pub symbol_map: HashMap<String, Symbol>,
}

#[derive(thiserror::Error, Debug)]
pub enum CompilationError {
    #[error("undefined symbol: {0:?}")]
    UndefinedSymbol(String),
    #[error("invalid state space")]
    InvalidStateSpace,
    #[error("invalid operand: {0:?}")]
    InvalidOperand(ast::Operand),
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
        let (mut frame_desc, instructions) = state.finalize();
        frame_desc.iptr = iptr;
        self.instructions.extend(instructions);
        todo!("register function descriptor, add to symbol map")
    }
}

struct FuncCodegenState<'a> {
    parent: &'a CompiledModule,
    instructions: Vec<vm::Instruction>,
    var_map: VariableMap,
    regs: vm::RegDesc,
    stack_size: usize,
    param_stack_offset: usize,
}

impl<'a> FuncCodegenState<'a> {
    pub fn new(parent: &'a CompiledModule) -> Self {
        Self {
            parent,
            instructions: Vec::new(),
            var_map: VariableMap::new(),
            regs: vm::RegDesc::default(),
            stack_size: 0,
            param_stack_offset: 0,
        }
    }

    fn declare_var(&mut self, decl: ast::VarDecl) -> Result<(), CompilationError> {
        use ast::StateSpace;
        use ast::Type::*;
        use vm::RegOperand;
        match decl.state_space {
            StateSpace::Register => {
                let opref = match decl.ty {
                    Bit128 => RegOperand::B128(self.regs.alloc_b128()),
                    Bit64 | Unsigned64 | Signed64 | Float64 => {
                        RegOperand::B64(self.regs.alloc_b64())
                    }
                    Bit32 | Unsigned32 | Signed32 | Float32 | Float16x2 => {
                        RegOperand::B32(self.regs.alloc_b32())
                    }
                    Bit16 | Unsigned16 | Signed16 | Float16 => {
                        RegOperand::B16(self.regs.alloc_b16())
                    }
                    Bit8 | Unsigned8 | Signed8 => RegOperand::B8(self.regs.alloc_b8()),
                    Predicate => RegOperand::Pred(self.regs.alloc_pred()),
                };
                self.var_map.insert(decl.ident, Variable::Register(opref));
            }
            StateSpace::Global => todo!(),
            StateSpace::Local => todo!(),
            StateSpace::Shared => todo!(),
            StateSpace::Constant => todo!(),
            StateSpace::Parameter => todo!(),
        }
        Ok(())
    }

    fn handle_vars(&mut self, vars: Vec<ast::VarDecl>) -> Result<(), CompilationError> {
        for decl in vars {
            if let Some(mult) = decl.multiplicity {
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
            if matches!(param.ty, ast::Type::Predicate) {
                // this should raise an error as predicates can only exist in the reg state space
                todo!()
            }
            // align to required alignment
            assert!(param.ty.alignment().count_ones() == 1);
            self.param_stack_offset =
                (self.param_stack_offset + param.ty.alignment() - 1) & !(param.ty.alignment() - 1);

            let loc = vm::AddrOperand::StackRelative(-(self.param_stack_offset as isize));

            // account for the size of the parameter
            self.param_stack_offset += param.ty.size();

            self.var_map
                .insert(param.ident.clone(), Variable::Memory(loc));
        }
        Ok(())
    }

    fn construct_immediate(&mut self, ty: vm::Type, imm: i32) -> Result<vm::RegOperand, CompilationError> {
        todo!()
    }

    fn get_src_reg(&mut self, ty: vm::Type, op: &ast::Operand) -> Result<vm::RegOperand, CompilationError> {
        use ast::Operand;
        match op {
            Operand::Variable(ident) => self.var_map.get_reg(ident),
            Operand::Immediate(imm) => self.construct_immediate(ty, *imm),
            Operand::SpecialReg(special) => Ok(vm::RegOperand::Special(*special)),
            Operand::Address(_) => todo!(),
        }
    }

    fn handle_instruction(&mut self, instr: ast::Instruction) -> Result<(), CompilationError> {
        use vm::AddrOperand;
        use ast::AddressOperand;
        use ast::InstructionSpecifier::*;
        use ast::Operand;
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
                        let vm::RegOperand::B64(reg) = reg else {
                            // can only use 64-bit registers as addresses
                            todo!()
                        };
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
                        let vm::RegOperand::B64(reg) = reg else {
                            // can only use 64-bit registers as addresses
                            todo!()
                        };
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
                let [Operand::Variable(dst), src_op] = instr.operands.as_slice() else {
                    todo!()
                };
                let ty = ty.to_vm();
                let src_reg = self.get_src_reg(ty, src_op)?;
                let dst_reg = self.var_map.get_reg(dst)?;
                self.instructions
                    .push(vm::Instruction::Move(ty, dst_reg, src_reg));
            }
            Add(ty) => {
                let [Operand::Variable(dst), lhs_op, rhs_op] =
                    instr.operands.as_slice()
                else {
                    todo!()
                };
                let ty = ty.to_vm();
                let dst_reg = self.var_map.get_reg(dst)?;
                let lhs_reg = self.get_src_reg(ty, lhs_op)?;
                let rhs_reg = self.get_src_reg(ty, rhs_op)?;
                self.instructions
                    .push(vm::Instruction::Add(ty, dst_reg, lhs_reg, rhs_reg));
            }
            Multiply(mode, ty) => {
                let [Operand::Variable(dst), lhs_op, rhs_op] =
                    instr.operands.as_slice()
                else {
                    todo!()
                };
                let ty = ty.to_vm();
                let dst_reg = self.var_map.get_reg(dst)?;
                let lhs_reg = self.get_src_reg(ty, lhs_op)?;
                let rhs_reg = self.get_src_reg(ty, rhs_op)?;
                self.instructions.push(vm::Instruction::Mul(
                    ty,
                    mode,
                    dst_reg,
                    lhs_reg,
                    rhs_reg,
                ));
            }
            MultiplyAdd(_, _) => todo!(),
            Convert { from, to } => todo!(),
            ConvertAddress(ty, st) => {
                // for now, just move the address register into the destination register
                let [Operand::Variable(dst), Operand::Variable(src)] = instr.operands.as_slice()
                else {
                    todo!();
                };
                let dst_reg = self.var_map.get_reg(dst)?;
                let src_reg = self.var_map.get_reg(src)?;
                self.instructions
                    .push(vm::Instruction::Move(ty.to_vm(), dst_reg, src_reg));
            }
            ConvertAddressTo(_, _) => todo!(),
            SetPredicate(_, _) => todo!(),
            ShiftLeft(_) => todo!(),
            Call {
                uniform,
                ident,
                ret_param,
                params,
            } => todo!(),
            BarrierSync => todo!(),
            Branch => todo!(),
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

    pub fn finalize(self) -> (vm::FuncFrameDesc, Vec<vm::Instruction>) {
        // align stack size to 16 bytes
        let aligned_stack = (self.stack_size + 15) & !15;
        let frame_desc = vm::FuncFrameDesc {
            iptr: vm::IPtr(self.instructions.len()),
            frame_size: aligned_stack,
            arg_size: self.param_stack_offset,
            regs: self.regs,
        };
        (frame_desc, self.instructions)
    }
}

pub fn compile(module: ast::Module) -> Result<CompiledModule, CompilationError> {
    let mut cmod = CompiledModule {
        instructions: Vec::new(),
        func_descriptors: Vec::new(),
        global_vars: Vec::new(),
        symbol_map: HashMap::new(),
    };
    for directive in module.0 {
        cmod.compile_directive_toplevel(directive)?;
    }
    Ok(cmod)
}
