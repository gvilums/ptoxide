pub mod compiler;
pub mod vm;

use std::fs;

use logos::{Lexer, Logos};
use thiserror::Error;

fn lex_reg_multiplicity(lex: &mut Lexer<Token>) -> Option<u32> {
    let mut s = lex.slice();
    s = &s[1..s.len() - 1];
    s.parse().ok()
}

fn lex_version_number(lex: &mut Lexer<Token>) -> Option<Version> {
    let (major_str, minor_str) = lex.slice().split_once('.')?;
    let major = major_str.parse().ok()?;
    let minor = minor_str.parse().ok()?;
    Some(Version { major, minor })
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct LexError;

impl std::fmt::Display for LexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", LexError)
    }
}
impl std::error::Error for LexError {}

#[derive(Logos, Debug, PartialEq, Clone)]
#[logos(skip r"[ \t\n\f]+")] // Ignore this regex pattern between tokens
#[logos(error = LexError)]
pub enum Token {
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
    #[token(".align")]
    Align,

    #[token(".b128")]
    Bit128,
    #[token(".b64")]
    Bit64,
    #[token(".b32")]
    Bit32,
    #[token(".b16")]
    Bit16,
    #[token(".b8")]
    Bit8,
    #[token(".u64")]
    Unsigned64,
    #[token(".u32")]
    Unsigned32,
    #[token(".u16")]
    Unsigned16,
    #[token(".u8")]
    Unsigned8,
    #[token(".s64")]
    Signed64,
    #[token(".s32")]
    Signed32,
    #[token(".s16")]
    Signed16,
    #[token(".s8")]
    Signed8,
    #[token(".f64")]
    Float64,
    #[token(".f32")]
    Float32,
    #[token(".f16x2")]
    Float16x2,
    #[token(".f16")]
    Float16,
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
    #[token("mul")]
    Mul,
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
    #[token("call")]
    Call,

    #[token("bar")]
    #[token("barrier")]
    Bar,

    #[token(".cta")]
    Cta,

    #[token(".sync")]
    Sync,

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

    #[token(".uni")]
    Uniform,

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

    #[regex(r"<\s*\+?\d+\s*>", lex_reg_multiplicity)]
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

    #[regex(r"\d+\.\d+", lex_version_number)]
    VersionNumber(Version),

    #[regex(r"//.*", logos::skip)]
    Skip,
}

impl Token {
    fn is_directive(&self) -> bool {
        matches!(
            self,
            Token::Version
                | Token::Target
                | Token::AddressSize
                | Token::Visible
                | Token::Entry
                | Token::Param
                | Token::Reg
                | Token::Global
                | Token::Local
                | Token::Shared
                | Token::Const
                | Token::Align,
        )
    }
}

#[derive(Error, Debug)]
pub enum ParseErr {
    #[error("Unexpected token: {:?}", .0)]
    UnexpectedToken(Token),
    #[error("Unexpected end of file")]
    UnexpectedEof,
    #[error("Lex error")]
    LexError(#[from] LexError),
}

type ParseResult<'a, T> = Result<(T, Scanner<'a>), ParseErr>;

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

    fn must_get(&self) -> Result<&'a Token, ParseErr> {
        self.get().ok_or(ParseErr::UnexpectedEof)
    }

    fn skip(&mut self) {
        if self.tokens.len() > 0 {
            self.tokens = &self.tokens[1..];
        }
    }

    fn consume(&mut self, token: Token) -> Result<(), ParseErr> {
        let head = self.get().ok_or(ParseErr::UnexpectedEof)?;
        if head == &token {
            self.skip();
            Ok(())
        } else {
            Err(ParseErr::UnexpectedToken(head.clone()))
        }
    }

    fn pop(&mut self) -> Option<&'a Token> {
        let head = self.get();
        self.skip();
        head
    }

    fn must_pop(&mut self) -> Result<&'a Token, ParseErr> {
        self.pop().ok_or(ParseErr::UnexpectedEof)
    }
}

type Ident = String;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Version {
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
    body: Statement,
}

#[derive(Debug)]
struct FunctionParam {
    ident: Ident,
    ty: Type,
    alignment: Option<u32>,
    array_bounds: Vec<u32>,
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
    Bit128,
    Bit64,
    Bit32,
    Bit16,
    Bit8,
    Unsigned64,
    Unsigned32,
    Unsigned16,
    Unsigned8,
    Signed64,
    Signed32,
    Signed16,
    Signed8,
    Float64,
    Float32,
    Float16x2,
    Float16,
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
    alignment: Option<u32>,
    array_bounds: Vec<u32>,
    multiplicity: Option<u32>,
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
enum Directive {
    VarDecl(Variable),
}

#[derive(Debug)]
struct Instruction {
    label: Option<Ident>,
    guard: Option<Guard>,
    specifier: InstructionSpecifier,
    operands: Vec<Operand>,
}

#[derive(Debug)]
enum Statement {
    Directive(Directive),
    Instruction(Instruction),
    Grouping(Vec<Statement>),
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
    Multiply(Mode, Type),
    MultiplyAdd(Mode, Type),
    Convert { from: Type, to: Type },
    ConvertAddress(Type, StateSpace),
    ConvertAddressTo(Type, StateSpace),
    SetPredicate(Predicate, Type),
    ShiftLeft(Type),
    BarrierSync,
    Branch,
    Return,
}

fn parse_program(src: &str) -> Result<Module, ParseErr> {
    let tokens = Token::lexer(src).collect::<Result<Vec<_>, _>>()?;
    let scanner = Scanner::new(&tokens);
    let (module, scanner) = parse_module(scanner)?;
    match scanner.get() {
        Some(_) => match parse_function(scanner) {
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
    match scanner.must_pop()? {
        Token::VersionNumber(version) => Ok((*version, scanner)),
        t => Err(ParseErr::UnexpectedToken(t.clone())),
    }
}

fn parse_target(mut scanner: Scanner) -> ParseResult<String> {
    scanner.consume(Token::Target)?;
    let target = scanner.pop().ok_or(ParseErr::UnexpectedEof)?;
    let Token::Identifier(target) = target else {
        return Err(ParseErr::UnexpectedToken(target.clone()));
    };
    Ok((target.clone(), scanner))
}

fn parse_address_size(mut scanner: Scanner) -> ParseResult<AddressSize> {
    scanner.consume(Token::AddressSize)?;
    let size = scanner.pop().ok_or(ParseErr::UnexpectedEof)?;
    let Token::Integer(size) = size else {
        return Err(ParseErr::UnexpectedToken(size.clone()));
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
            Some(Token::LeftBracket) => scanner.skip(),
            _ => break Ok((bounds, scanner)),
        }
        let Token::Integer(bound) = scanner.must_pop()? else {
            return Err(ParseErr::UnexpectedToken(scanner.must_get()?.clone()));
        };
        scanner.consume(Token::RightBracket)?;
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
        t => Err(ParseErr::UnexpectedToken(t.clone())),
    }
}

fn parse_alignment(mut scanner: Scanner) -> ParseResult<u32> {
    scanner.consume(Token::Align)?;
    let alignment = match scanner.must_pop()? {
        Token::Integer(i) => *i as u32,
        t => return Err(ParseErr::UnexpectedToken(t.clone())),
    };
    Ok((alignment, scanner))
}

fn parse_type(mut scanner: Scanner) -> ParseResult<Type> {
    let ty = match scanner.must_pop()? {
        Token::Bit8 => Type::Bit8,
        Token::Bit16 => Type::Bit16,
        Token::Bit32 => Type::Bit32,
        Token::Bit64 => Type::Bit64,
        Token::Bit128 => Type::Bit128,
        Token::Unsigned8 => Type::Unsigned8,
        Token::Unsigned16 => Type::Unsigned16,
        Token::Unsigned32 => Type::Unsigned32,
        Token::Unsigned64 => Type::Unsigned64,
        Token::Signed8 => Type::Signed8,
        Token::Signed16 => Type::Signed16,
        Token::Signed32 => Type::Signed32,
        Token::Signed64 => Type::Signed64,
        Token::Float16 => Type::Float16,
        Token::Float16x2 => Type::Float16x2,
        Token::Float32 => Type::Float32,
        Token::Float64 => Type::Float64,
        Token::Predicate => Type::Predicate,
        t => return Err(ParseErr::UnexpectedToken(t.clone())),
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
        t => return Err(ParseErr::UnexpectedToken(t.clone())),
    };
    Ok((reg, scanner))
}

fn parse_mode(mut scanner: Scanner) -> ParseResult<Mode> {
    let mode = match scanner.must_pop()? {
        Token::Low => Mode::Low,
        Token::High => Mode::High,
        Token::Wide => Mode::Wide,
        t => return Err(ParseErr::UnexpectedToken(t.clone())),
    };
    Ok((mode, scanner))
}

fn parse_variable(scanner: Scanner) -> ParseResult<Variable> {
    let (state_space, mut scanner) = parse_state_space(scanner)?;

    let alignment = parse_alignment(scanner)
        .map(|(alignment, res)| {
            scanner = res;
            alignment
        })
        .ok();

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
        t => return Err(ParseErr::UnexpectedToken(t.clone())),
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
            alignment,
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
                t => return Err(ParseErr::UnexpectedToken(t.clone())),
            };
            Guard::Negated(ident)
        }
        t => return Err(ParseErr::UnexpectedToken(t.clone())),
    };
    Ok((guard, scanner))
}

fn parse_predicate(mut scanner: Scanner) -> ParseResult<Predicate> {
    let pred = match scanner.must_pop()? {
        Token::Ge => Predicate::GreaterThan,
        t => return Err(ParseErr::UnexpectedToken(t.clone())),
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
        Token::Mul => {
            let (mode, scanner) = parse_mode(scanner)?;
            let (ty, scanner) = parse_type(scanner)?;
            Ok((InstructionSpecifier::Multiply(mode, ty), scanner))
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
        Token::ConvertAddress => match scanner.get() {
            Some(Token::To) => {
                scanner.skip();
                let (state_space, scanner) = parse_state_space(scanner)?;
                let (ty, scanner) = parse_type(scanner)?;
                Ok((
                    InstructionSpecifier::ConvertAddressTo(ty, state_space),
                    scanner,
                ))
            }
            _ => {
                let (state_space, scanner) = parse_state_space(scanner)?;
                let (ty, scanner) = parse_type(scanner)?;
                Ok((
                    InstructionSpecifier::ConvertAddress(ty, state_space),
                    scanner,
                ))
            }
        },
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
        Token::Bar => {
            // cta token is meaningless
            if let Some(Token::Cta) = scanner.get() {
                scanner.skip();
            }
            scanner.consume(Token::Sync)?;
            Ok((InstructionSpecifier::BarrierSync, scanner))
        }
        t => Err(ParseErr::UnexpectedToken(t.clone())),
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
                    t => return Err(ParseErr::UnexpectedToken(t.clone())),
                };
                scanner.consume(Token::RightBracket)?;
                Operand::Address(ident)
            }
            t => return Err(ParseErr::UnexpectedToken(t.clone())),
        };
        operands.push(operand);
    }
}

fn parse_grouping(mut scanner: Scanner) -> ParseResult<Vec<Statement>> {
    scanner.consume(Token::LeftBrace)?; // Consume the left brace
    let mut statements = Vec::new();
    loop {
        if let Some(Token::RightBrace) = scanner.get() {
            scanner.skip();
            break Ok((statements, scanner));
        }
        match parse_statement(scanner) {
            Ok((basic_block, rest)) => {
                statements.push(basic_block);
                scanner = rest;
            }
            Err(e) => break Err(e),
        }
    }
}

fn parse_directive(scanner: Scanner) -> ParseResult<Directive> {
    let (var, scanner) = parse_variable(scanner)?;
    Ok((Directive::VarDecl(var), scanner))
}

fn parse_instruction(mut scanner: Scanner) -> ParseResult<Instruction> {
    let label = if let Some(Token::Identifier(s)) = scanner.get() {
        let s = s.clone();
        scanner.skip();
        scanner.consume(Token::Colon)?;
        Some(s)
    } else {
        None
    };
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
            label,
            guard,
            specifier,
            operands,
        },
        scanner,
    ))
}

fn parse_statement(scanner: Scanner) -> ParseResult<Statement> {
    match scanner.must_get()? {
        Token::LeftBrace => {
            let (grouping, scanner) = parse_grouping(scanner)?;
            Ok((Statement::Grouping(grouping), scanner))
        }
        t if t.is_directive() => {
            let (dir, scanner) = parse_directive(scanner)?;
            Ok((Statement::Directive(dir), scanner))
        }
        _ => {
            let (instr, scanner) = parse_instruction(scanner)?;
            Ok((Statement::Instruction(instr), scanner))
        }
    }
}

fn parse_function_param(mut scanner: Scanner) -> ParseResult<FunctionParam> {
    scanner.consume(Token::Param)?; // Consume the param keyword

    let alignment = parse_alignment(scanner)
        .map(|(alignment, res)| {
            scanner = res;
            alignment
        })
        .ok();

    let (ty, mut scanner) = parse_type(scanner)?;
    let ident = loop {
        match scanner.pop() {
            Some(Token::Identifier(s)) => break s.clone(),
            // Some(token) => return Err(ParseErr::UnexpectedToken(token)),
            Some(_) => {}
            None => return Err(ParseErr::UnexpectedEof),
        }
    };

    let (array_bounds, mut scanner) = parse_array_bounds(scanner)?;

    let fparam = FunctionParam {
        alignment,
        ident,
        ty,
        array_bounds,
    };
    match scanner.get() {
        Some(Token::Comma) => {
            scanner.skip();
            Ok((fparam, scanner))
        }
        Some(Token::RightParen) => Ok((fparam, scanner)),
        Some(token) => Err(ParseErr::UnexpectedToken(token.clone())),
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
            Some(token) => return Err(ParseErr::UnexpectedToken(token.clone())),
            None => return Err(ParseErr::UnexpectedEof),
        }
    };
    let (params, scanner) = parse_function_params(scanner)?;
    let (body, scanner) = parse_statement(scanner)?;

    Ok((
        Function {
            ident,
            visible,
            entry,
            params,
            body,
        },
        scanner,
    ))
}

// generic address is 64 bits, of which the upper 3 bits denote the space
// the lower 61 bits denote the address in that space

// addresses specific to a space _do not_ have the space bits set.
// instead, they are simply offsets into the space

// memory layout design
// each state space is repsented in the virtual machine as a contiguous block of memory
// allocated in a rust-native structure, such as a Vec
//
// the exception to this is the register space. registers of each type are allocated
//

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_parse_add() {
        let contents = fs::read_to_string("kernels/add.ptx").unwrap();
        let _ = parse_program(&contents).unwrap();
    }

    #[test]
    fn test_parse_transpose() {
        let contents = fs::read_to_string("kernels/transpose.ptx").unwrap();
        let _ = parse_program(&contents).unwrap();
    }

    #[test]
    fn test_parse_add_simple() {
        let contents = fs::read_to_string("kernels/add_simple.ptx").unwrap();
        let _ = parse_program(&contents).unwrap();
    }

    #[test]
    fn test_parse_test() {
        let contents = fs::read_to_string("kernels/test.ptx").unwrap();
        let _ = parse_program(&contents).unwrap();
    }
}
