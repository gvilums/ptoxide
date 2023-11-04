use std::fs;

use logos::{Lexer, Logos};
use thiserror::Error;

fn reg_multiplicity(lex: &mut Lexer<Token>) -> Option<u32> {
    let mut s = lex.slice();
    s = &s[1..s.len() - 1];
    s.parse().ok()
}

#[derive(Logos, Debug, PartialEq, Clone)]
#[logos(skip r"[ \t\n\f]+")] // Ignore this regex pattern between tokens
enum Token {
    #[token(".version")]
    Version,
    #[token(".target")]
    Target,
    #[token(".address_size")]
    AddressSize,
    #[token(".visible")]
    Visible,
    #[token(".entry")]
    Entry,
    #[token(".param")]
    Param,
    #[token(".reg")]
    Reg,
    #[token(".global")]
    Global,
    #[token(".local")]
    Local,
    #[token(".shared")]
    Shared,
    #[token(".const")]
    Const,

    #[token(".b64")]
    Bit64,
    #[token(".b32")]
    Bit32,
    #[token(".u64")]
    Unsigned64,
    #[token(".u32")]
    Unsigned32,
    #[token(".s64")]
    Signed64,
    #[token(".s32")]
    Signed32,
    #[token(".f64")]
    Float64,
    #[token(".f32")]
    Float32,
    #[token(".pred")]
    Predicate,

    #[token(".v2")]
    V2,
    #[token(".v4")]
    V4,

    #[token("ld")]
    Load,
    #[token("st")]
    Store,
    #[token("add")]
    Add,
    #[token("mov")]
    Move,
    #[token("mad")]
    MultiplyAdd,
    #[token("shl")]
    ShiftLeft,
    #[token("cvt")]
    Convert,
    #[token("cvta")]
    ConvertAddress,
    #[token("ret")]
    Return,
    #[token("bra")]
    Branch,
    #[token("setp")]
    SetPredicate,

    #[token(".to")]
    To,

    #[token(".lo")]
    Low,
    #[token(".hi")]
    High,
    #[token(".wide")]
    Wide,

    #[token(".ge")]
    Ge,

    #[token("%tid")]
    ThreadId,
    #[token("%tid.x")]
    ThreadIdX,
    #[token("%tid.y")]
    ThreadIdY,
    #[token("%tid.z")]
    ThreadIdZ,

    #[token("%ntid")]
    NumThreads,
    #[token("%ntid.x")]
    NumThreadsX,
    #[token("%ntid.y")]
    NumThreadsY,
    #[token("%ntid.z")]
    NumThreadsZ,

    #[token("%ctaid")]
    CtaId,
    #[token("%ctaid.x")]
    CtaIdX,
    #[token("%ctaid.y")]
    CtaIdY,
    #[token("%ctaid.z")]
    CtaIdZ,

    #[regex(r"[a-zA-Z][a-zA-Z0-9_$]*|[_$%][a-zA-Z0-9_$]+", |lex| lex.slice().to_string())]
    Identifier(String),

    #[regex(r"[0-9]+", |lex| lex.slice().parse().ok())]
    Integer(i32),

    #[regex(r"<\s*\+?\d+\s*>", reg_multiplicity)]
    RegMultiplicity(u32),

    #[token("{")]
    LeftBrace,
    #[token("}")]
    RightBrace,
    #[token("(")]
    LeftParen,
    #[token(")")]
    RightParen,
    #[token("[")]
    LeftBracket,
    #[token("]")]
    RightBracket,
    #[token("@")]
    At,
    #[token("!")]
    Exclamation,
    #[token(";")]
    Semicolon,
    #[token(":")]
    Colon,
    #[token(",")]
    Comma,
    #[token(".")]
    Dot,

    #[regex(r"//.*", logos::skip)]
    Skip,
}

#[derive(Error, Debug)]
enum ParseErr<'a> {
    #[error("Unexpected token: {:?}", .0)]
    UnexpectedToken(&'a Token),
    #[error("Unexpected end of file")]
    UnexpectedEof,
}

type ParseResult<'a, T> = Result<(T, Scanner<'a>), ParseErr<'a>>;

#[derive(Clone, Copy, Debug)]
struct Scanner<'a> {
    tokens: &'a [Token],
}

impl<'a> Scanner<'a> {
    fn new(tokens: &'a [Token]) -> Self {
        Scanner { tokens }
    }

    fn get(&self) -> Option<&'a Token> {
        self.tokens.first()
    }

    fn must_get(&self) -> Result<&'a Token, ParseErr<'a>> {
        self.get().ok_or(ParseErr::UnexpectedEof)
    }

    fn skip(&mut self) {
        if self.tokens.len() > 0 {
            self.tokens = &self.tokens[1..];
        }
    }

    fn consume(&mut self, token: Token) -> Result<(), ParseErr<'a>> {
        let head = self.get().ok_or(ParseErr::UnexpectedEof)?;
        if head == &token {
            self.skip();
            Ok(())
        } else {
            Err(ParseErr::UnexpectedToken(head))
        }
    }

    fn pop(&mut self) -> Option<&'a Token> {
        let head = self.get();
        self.skip();
        head
    }

    fn must_pop(&mut self) -> Result<&'a Token, ParseErr<'a>> {
        self.pop().ok_or(ParseErr::UnexpectedEof)
    }
}

type Ident = String;

#[derive(Debug)]
struct Version {
    major: i32,
    minor: i32,
}

#[derive(Debug)]
enum AddressSize {
    Adr32,
    Adr64,
    Other,
}

#[derive(Debug)]
struct Module {
    version: Version,
    target: String,
    addr_size: AddressSize,

    functions: Vec<Function>,
}

#[derive(Debug)]
struct Function {
    ident: Ident,
    visible: bool,
    entry: bool,
    params: Vec<FunctionParam>,
    basic_blocks: Vec<BasicBlock>,
}

#[derive(Debug)]
struct FunctionParam {
    ident: Ident,
}

#[derive(Debug)]
enum StateSpace {
    Global,
    Local,
    Shared,
    Register,
    Constant,
    Parameter,
}

#[derive(Debug)]
enum Type {
    Bit64,
    Bit32,
    Unsigned64,
    Unsigned32,
    Signed64,
    Signed32,
    Float64,
    Float32,
    Predicate,
}

#[derive(Debug)]
enum Vector {
    V2,
    V4,
}

#[derive(Debug)]
enum SpecialReg {
    ThreadId,
    ThreadIdX,
    ThreadIdY,
    ThreadIdZ,
    NumThreads,
    NumThreadsX,
    NumThreadsY,
    NumThreadsZ,
    CtaId,
    CtaIdX,
    CtaIdY,
    CtaIdZ,
}

#[derive(Debug)]
struct Variable {
    state_space: StateSpace,
    ty: Type,
    vector: Option<Vector>,
    ident: Ident,
    array_bounds: Vec<u32>,
    multiplicity: Option<u32>,
}

#[derive(Debug)]
struct BasicBlock {
    label: Option<Ident>,
    variables: Vec<Variable>,
    instructions: Vec<Instruction>,
}

#[derive(Debug)]
enum Operand {
    SpecialReg(SpecialReg),
    Identifier(Ident),
    Immediate(i32),
    Address(Ident),
}

#[derive(Debug)]
enum Guard {
    Normal(Ident),
    Negated(Ident),
}

#[derive(Debug)]
struct Instruction {
    guard: Option<Guard>,
    specifier: InstructionSpecifier,
    operands: Vec<Operand>,
}

#[derive(Debug)]
enum Predicate {
    LessThan,
    GreaterThan,
}

#[derive(Debug)]
enum Mode {
    Low,
    High,
    Wide,
}

#[derive(Debug)]
enum InstructionSpecifier {
    Load(StateSpace, Type),
    Store(StateSpace, Type),
    Move(Type),
    Add(Type),
    MultiplyAdd(Mode, Type),
    Convert { from: Type, to: Type },
    ConvertAddress(Type, StateSpace),
    ConvertAddressTo(Type, StateSpace),
    SetPredicate(Predicate, Type),
    ShiftLeft(Type),
    Branch,
    Return,
}

fn parse_program(scanner: Scanner) -> Result<Module, ParseErr> {
    let (module, scanner) = parse_module(scanner)?;
    match scanner.get() {
        Some(token) => match parse_function(scanner) {
            Ok((function, _)) => {
                panic!(
                    "Function failed to parse in module but succeeded by itself: {:?}",
                    function
                );
            }
            Err(e) => Err(e),
        },
        None => Ok(module),
    }
}

fn parse_version(mut scanner: Scanner) -> ParseResult<Version> {
    scanner.consume(Token::Version)?;
    let major = scanner.pop().ok_or(ParseErr::UnexpectedEof)?;
    let Token::Integer(major) = major else {
        return Err(ParseErr::UnexpectedToken(major));
    };
    scanner.consume(Token::Dot)?;
    let minor = scanner.pop().ok_or(ParseErr::UnexpectedEof)?;
    let Token::Integer(minor) = minor else {
        return Err(ParseErr::UnexpectedToken(minor));
    };
    Ok((
        Version {
            major: *major,
            minor: *minor,
        },
        scanner,
    ))
}

fn parse_target(mut scanner: Scanner) -> ParseResult<String> {
    scanner.consume(Token::Target)?;
    let target = scanner.pop().ok_or(ParseErr::UnexpectedEof)?;
    let Token::Identifier(target) = target else {
        return Err(ParseErr::UnexpectedToken(target));
    };
    Ok((target.clone(), scanner))
}

fn parse_address_size(mut scanner: Scanner) -> ParseResult<AddressSize> {
    scanner.consume(Token::AddressSize)?;
    let size = scanner.pop().ok_or(ParseErr::UnexpectedEof)?;
    let Token::Integer(size) = size else {
        return Err(ParseErr::UnexpectedToken(size));
    };
    match size {
        32 => Ok((AddressSize::Adr32, scanner)),
        64 => Ok((AddressSize::Adr64, scanner)),
        _ => Ok((AddressSize::Other, scanner)),
    }
}

fn parse_module(scanner: Scanner) -> ParseResult<Module> {
    let (version, scanner) = parse_version(scanner)?;
    let (target, scanner) = parse_target(scanner)?;
    let (addr_size, scanner) = parse_address_size(scanner)?;
    let (functions, scanner) = parse_functions(scanner)?;
    Ok((
        Module {
            version,
            target,
            addr_size,
            functions,
        },
        scanner,
    ))
}

fn parse_functions(mut scanner: Scanner) -> ParseResult<Vec<Function>> {
    let mut functions = Vec::new();
    loop {
        match parse_function(scanner) {
            Ok((function, rest)) => {
                functions.push(function);
                scanner = rest;
            }
            Err(_) => break Ok((functions, scanner)),
        }
    }
}

fn parse_array_bounds(mut scanner: Scanner) -> ParseResult<Vec<u32>> {
    let mut bounds = Vec::new();
    loop {
        match scanner.get() {
            Some(Token::RightBracket) => scanner.skip(),
            _ => break Ok((bounds, scanner)),
        }
        let Token::Integer(bound) = scanner.must_pop()? else {
            return Err(ParseErr::UnexpectedToken(scanner.must_get()?));
        };
        scanner.consume(Token::LeftBracket)?;
        // todo clean up raw casts
        bounds.push(*bound as u32);
    }
}

fn parse_state_space(mut scanner: Scanner) -> ParseResult<StateSpace> {
    match scanner.must_pop()? {
        Token::Global => Ok((StateSpace::Global, scanner)),
        Token::Local => Ok((StateSpace::Local, scanner)),
        Token::Shared => Ok((StateSpace::Shared, scanner)),
        Token::Reg => Ok((StateSpace::Register, scanner)),
        Token::Param => Ok((StateSpace::Parameter, scanner)),
        Token::Const => Ok((StateSpace::Constant, scanner)),
        t => Err(ParseErr::UnexpectedToken(t)),
    }
}

fn parse_type(mut scanner: Scanner) -> ParseResult<Type> {
    let ty = match scanner.must_pop()? {
        Token::Bit32 => Type::Bit32,
        Token::Bit64 => Type::Bit64,
        Token::Unsigned32 => Type::Unsigned32,
        Token::Unsigned64 => Type::Unsigned64,
        Token::Signed32 => Type::Signed32,
        Token::Signed64 => Type::Signed64,
        Token::Float32 => Type::Float32,
        Token::Float64 => Type::Float64,
        Token::Predicate => Type::Predicate,
        t => return Err(ParseErr::UnexpectedToken(t)),
    };
    Ok((ty, scanner))
}

fn parse_special_reg(mut scanner: Scanner) -> ParseResult<SpecialReg> {
    let reg = match scanner.must_pop()? {
        Token::ThreadId => SpecialReg::ThreadId,
        Token::ThreadIdX => SpecialReg::ThreadIdX,
        Token::ThreadIdY => SpecialReg::ThreadIdY,
        Token::ThreadIdZ => SpecialReg::ThreadIdZ,
        Token::NumThreads => SpecialReg::NumThreads,
        Token::NumThreadsX => SpecialReg::NumThreadsX,
        Token::NumThreadsY => SpecialReg::NumThreadsY,
        Token::NumThreadsZ => SpecialReg::NumThreadsZ,
        Token::CtaId => SpecialReg::CtaId,
        Token::CtaIdX => SpecialReg::CtaIdX,
        Token::CtaIdY => SpecialReg::CtaIdY,
        Token::CtaIdZ => SpecialReg::CtaIdZ,
        t => return Err(ParseErr::UnexpectedToken(t)),
    };
    Ok((reg, scanner))
}

fn parse_mode(mut scanner: Scanner) -> ParseResult<Mode> {
    let mode = match scanner.must_pop()? {
        Token::Low => Mode::Low,
        Token::High => Mode::High,
        Token::Wide => Mode::Wide,
        t => return Err(ParseErr::UnexpectedToken(t)),
    };
    Ok((mode, scanner))
}

fn parse_variable(scanner: Scanner) -> ParseResult<Variable> {
    let (state_space, mut scanner) = parse_state_space(scanner)?;

    let vector = match scanner.get() {
        Some(Token::V2) => {
            scanner.skip();
            Some(Vector::V2)
        }
        Some(Token::V4) => {
            scanner.skip();
            Some(Vector::V4)
        }
        _ => None,
    };

    let (ty, mut scanner) = parse_type(scanner)?;

    let ident = match scanner.must_pop()? {
        Token::Identifier(s) => s.clone(),
        t => return Err(ParseErr::UnexpectedToken(t)),
    };

    let multiplicity = match scanner.get() {
        Some(Token::RegMultiplicity(m)) => {
            scanner.skip();
            Some(*m)
        }
        _ => None,
    };

    let (array_bounds, mut scanner) = parse_array_bounds(scanner)?;

    scanner.consume(Token::Semicolon)?;

    Ok((
        Variable {
            state_space,
            ty,
            vector,
            array_bounds,
            ident,
            multiplicity,
        },
        scanner,
    ))
}

fn parse_guard(mut scanner: Scanner) -> ParseResult<Guard> {
    scanner.consume(Token::At)?;
    let guard = match scanner.must_pop()? {
        Token::Identifier(s) => Guard::Normal(s.clone()),
        Token::Exclamation => {
            let ident = match scanner.must_pop()? {
                Token::Identifier(s) => s.clone(),
                t => return Err(ParseErr::UnexpectedToken(t)),
            };
            Guard::Negated(ident)
        }
        t => return Err(ParseErr::UnexpectedToken(t)),
    };
    Ok((guard, scanner))
}

fn parse_predicate(mut scanner: Scanner) -> ParseResult<Predicate> {
    let pred = match scanner.must_pop()? {
        Token::Ge => Predicate::GreaterThan,
        t => return Err(ParseErr::UnexpectedToken(t)),
    };
    Ok((pred, scanner))
}

fn parse_instr_specifier(mut scanner: Scanner) -> ParseResult<InstructionSpecifier> {
    match scanner.must_pop()? {
        Token::Load => {
            let (state_space, scanner) = parse_state_space(scanner)?;
            let (ty, scanner) = parse_type(scanner)?;
            Ok((InstructionSpecifier::Load(state_space, ty), scanner))
        }
        Token::Store => {
            let (state_space, scanner) = parse_state_space(scanner)?;
            let (ty, scanner) = parse_type(scanner)?;
            Ok((InstructionSpecifier::Store(state_space, ty), scanner))
        }
        Token::Move => {
            let (ty, scanner) = parse_type(scanner)?;
            Ok((InstructionSpecifier::Move(ty), scanner))
        }
        Token::Add => {
            let (ty, scanner) = parse_type(scanner)?;
            Ok((InstructionSpecifier::Add(ty), scanner))
        }
        Token::MultiplyAdd => {
            let (mode, scanner) = parse_mode(scanner)?;
            let (ty, scanner) = parse_type(scanner)?;
            Ok((InstructionSpecifier::MultiplyAdd(mode, ty), scanner))
        }
        Token::Convert => {
            let (to, scanner) = parse_type(scanner)?;
            let (from, scanner) = parse_type(scanner)?;
            Ok((InstructionSpecifier::Convert { to, from }, scanner))
        }
        Token::ConvertAddress => {
            match scanner.get() {
                Some(Token::To) => {
                    scanner.skip();
                    let (state_space, scanner) = parse_state_space(scanner)?;
                    let (ty, scanner) = parse_type(scanner)?;
                    Ok((InstructionSpecifier::ConvertAddressTo(ty, state_space), scanner))
                }
                _ => {
                    let (state_space, scanner) = parse_state_space(scanner)?;
                    let (ty, scanner) = parse_type(scanner)?;
                    Ok((InstructionSpecifier::ConvertAddress(ty, state_space), scanner))
                }
            }
        }
        Token::SetPredicate => {
            let (pred, scanner) = parse_predicate(scanner)?;
            let (ty, scanner) = parse_type(scanner)?;
            Ok((InstructionSpecifier::SetPredicate(pred, ty), scanner))
        }
        Token::ShiftLeft => {
            let (ty, scanner) = parse_type(scanner)?;
            Ok((InstructionSpecifier::ShiftLeft(ty), scanner))
        }
        Token::Branch => Ok((InstructionSpecifier::Branch, scanner)),
        Token::Return => Ok((InstructionSpecifier::Return, scanner)),
        t => Err(ParseErr::UnexpectedToken(t)),
    }
}

fn parse_operands(mut scanner: Scanner) -> ParseResult<Vec<Operand>> {
    let mut operands = Vec::new();
    loop {
        match scanner.get() {
            Some(Token::Semicolon) => {
                scanner.skip();
                break Ok((operands, scanner));
            }
            Some(Token::Comma) => scanner.skip(),
            _ => {}
        }
        // first try to parse a special register
        if let Ok((operand, rest)) = parse_special_reg(scanner) {
            scanner = rest;
            operands.push(Operand::SpecialReg(operand));
            continue;
        }
        // then fall back to other options
        let operand = match scanner.must_pop()? {
            Token::Identifier(s) => Operand::Identifier(s.clone()),
            Token::Integer(i) => Operand::Immediate(*i),
            Token::LeftBracket => {
                let ident = match scanner.must_pop()? {
                    Token::Identifier(s) => s.clone(),
                    t => return Err(ParseErr::UnexpectedToken(t)),
                };
                scanner.consume(Token::RightBracket)?;
                Operand::Address(ident)
            }
            t => return Err(ParseErr::UnexpectedToken(t)),
        };
        operands.push(operand);
    }
}

fn parse_instruction(mut scanner: Scanner) -> ParseResult<Instruction> {
    let guard = if let Ok((guard, res)) = parse_guard(scanner) {
        scanner = res;
        Some(guard)
    } else {
        None
    };

    let (specifier, scanner) = parse_instr_specifier(scanner)?;
    let (operands, scanner) = parse_operands(scanner)?;

    Ok((
        Instruction {
            guard,
            specifier,
            operands,
        },
        scanner,
    ))
}

fn parse_basic_block(mut scanner: Scanner) -> ParseResult<BasicBlock> {
    let label = match scanner.get() {
        Some(Token::Identifier(s)) => {
            scanner.skip();
            scanner.consume(Token::Colon)?;
            Some(s.clone())
        }
        _ => None,
    };
    let mut variables = Vec::new();
    let mut instructions = Vec::new();
    while let Some(t) = scanner.get() {
        // if our next token is a right brace or an identifier, this basic block is done
        if matches!(t, Token::RightBrace | Token::Identifier(_)) {
            break;
        }
        if let Ok((var, rest)) = parse_variable(scanner) {
            variables.push(var);
            scanner = rest;
        } else {
            let (inst, rest) = parse_instruction(scanner)?;
            instructions.push(inst);
            scanner = rest;
        }
    }
    Ok((
        BasicBlock {
            label,
            variables,
            instructions,
        },
        scanner,
    ))
}

fn parse_function_body(mut scanner: Scanner) -> ParseResult<Vec<BasicBlock>> {
    scanner.consume(Token::LeftBrace)?; // Consume the left brace
    let mut basic_blocks = Vec::new();
    loop {
        if let Some(Token::RightBrace) = scanner.get() {
            scanner.skip();
            break Ok((basic_blocks, scanner));
        }
        match parse_basic_block(scanner) {
            Ok((basic_block, rest)) => {
                basic_blocks.push(basic_block);
                scanner = rest;
            }
            Err(e) => break Err(e),
        }
    }
}

fn parse_function_param(mut scanner: Scanner) -> ParseResult<FunctionParam> {
    let ident = loop {
        match scanner.pop() {
            Some(Token::Identifier(s)) => break s.clone(),
            // Some(token) => return Err(ParseErr::UnexpectedToken(token)),
            Some(_) => {}
            None => return Err(ParseErr::UnexpectedEof),
        }
    };

    match scanner.get() {
        Some(Token::Comma) => {
            scanner.skip();
            Ok((FunctionParam { ident }, scanner))
        }
        Some(Token::RightParen) => Ok((FunctionParam { ident }, scanner)),
        Some(token) => Err(ParseErr::UnexpectedToken(token)),
        None => Err(ParseErr::UnexpectedEof),
    }
}

fn parse_function_params(mut scanner: Scanner) -> ParseResult<Vec<FunctionParam>> {
    scanner.consume(Token::LeftParen)?; // Consume the left paren
    let mut params = Vec::new();
    loop {
        match scanner.get() {
            Some(Token::RightParen) => {
                scanner.skip();
                break Ok((params, scanner));
            }
            Some(_) => {
                let (param, rest) = parse_function_param(scanner)?;
                params.push(param);
                scanner = rest;
            }
            None => return Err(ParseErr::UnexpectedEof),
        }
    }
}

fn parse_function(mut scanner: Scanner) -> ParseResult<Function> {
    let mut visible = false;
    let mut entry = false;
    let ident = loop {
        match scanner.pop() {
            Some(Token::Visible) => visible = true,
            Some(Token::Entry) => entry = true,
            Some(Token::Identifier(s)) => break s.clone(),
            Some(token) => return Err(ParseErr::UnexpectedToken(token)),
            None => return Err(ParseErr::UnexpectedEof),
        }
    };
    let (params, scanner) = parse_function_params(scanner)?;
    let (basic_blocks, scanner) = parse_function_body(scanner)?;

    Ok((
        Function {
            ident,
            visible,
            entry,
            params,
            basic_blocks,
        },
        scanner,
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let contents = fs::read_to_string("kernels/add.ptx")?;
    let Ok(tokens) = Token::lexer(&contents).collect::<Result<Vec<_>, _>>() else {
        panic!("Failed to lex file");
    };
    // dbg!(&tokens);
    let module = match parse_program(Scanner::new(&tokens)) {
        Ok(m) => m,
        Err(e) => panic!("Failed to parse file: {}", e),
    };
    dbg!(module);
    Ok(())
}
