use logos::{Lexer, Logos};
use thiserror::Error;

fn lex_reg_multiplicity(lex: &mut Lexer<Token>) -> Result<u32, LexError> {
    let mut s = lex.slice();
    s = &s[1..s.len() - 1];
    s.parse().map_err(|_| LexError::ParseRegMultiplicity)
}

fn lex_version_number(lex: &mut Lexer<Token>) -> Result<Version, LexError> {
    let Some((major_str, minor_str)) = lex.slice().split_once('.') else {
        return Err(LexError::ParseVersionNumber);
    };
    let major = major_str
        .parse()
        .map_err(|_| LexError::ParseVersionNumber)?;
    let minor = minor_str
        .parse()
        .map_err(|_| LexError::ParseVersionNumber)?;
    Ok(Version { major, minor })
}

fn lex_float32_constant(lex: &mut Lexer<Token>) -> Result<f32, LexError> {
    let Some(vals) = lex.slice().as_bytes().get(2..) else {
        return Err(LexError::ParseFloatConst);
    };
    let mut val = 0u32;
    for c in vals {
        val <<= 4;
        val |= match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            _ => return Err(LexError::ParseFloatConst),
        } as u32;
    }
    Ok(f32::from_bits(val))
}

fn lex_float64_constant(_lex: &mut Lexer<Token>) -> Option<f64> {
    todo!()
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum LexError {
    ParseFloatConst,
    ParseRegMultiplicity,
    ParseVersionNumber,
    #[default]
    Other,
}

impl std::fmt::Display for LexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
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
    #[token(".func")]
    Function,
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
    #[token(".noreturn")]
    NoReturn,
    #[token(".pragma")]
    Pragma,

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
    #[token("sub")]
    Sub,
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
    #[token("or")]
    Or,
    #[token("and")]
    And,
    #[token("fma")]
    FusedMulAdd,
    #[token("neg")]
    Negate,

    #[token("bar")]
    #[token("barrier")]
    Bar,

    #[token(".cta")]
    Cta,

    #[token(".sync")]
    Sync,

    #[token(".to")]
    To,

    #[token(".rn")]
    Rn,
    #[token(".rz")]
    Rz,
    #[token(".rm")]
    Rm,
    #[token(".rp")]
    Rp,

    #[token(".lo")]
    Low,
    #[token(".hi")]
    High,
    #[token(".wide")]
    Wide,

    #[token(".eq")]
    Eq,
    #[token(".ne")]
    Ne,
    #[token(".lt")]
    Lt,
    #[token(".le")]
    Le,
    #[token(".gt")]
    Gt,
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

    #[regex(r"-?[0-9]+", |lex| lex.slice().parse().ok(), priority=2)]
    IntegerConst(i64),
    // todo make sure this token does not conflict  with others
    // #[regex(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?", |lex| lex.slice().parse().ok())]
    #[regex(r"0[dD][0-9a-fA-F]{16}", lex_float64_constant)]
    Float64Const(f64),
    #[regex(r"0[fF][0-9a-fA-F]{8}", lex_float32_constant)]
    Float32Const(f32),

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
    #[token("+")]
    Plus,
    #[token(";")]
    Semicolon,
    #[token(":")]
    Colon,
    #[token(",")]
    Comma,

    #[regex(r#""[^"]*""#, |lex| lex.slice().to_string())]
    StringLiteral(String),
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
                | Token::Function
                | Token::Param
                | Token::Reg
                | Token::Global
                | Token::Local
                | Token::Shared
                | Token::Const
                | Token::Align
                | Token::Pragma
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

    fn consume_match(&mut self, token: Token) -> bool {
        let Some(head) = self.get() else {
            return false;
        };
        if head == &token {
            self.skip();
            true
        } else {
            false
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

#[derive(Clone, Debug)]
pub struct Pragma {
    value: String,
}

#[derive(Debug)]
pub enum AddressSize {
    Adr32,
    Adr64,
    Other,
}

#[derive(Debug)]
pub struct Module(pub Vec<Directive>);

#[derive(Debug)]
pub struct Function {
    pub ident: Ident,
    pub visible: bool,
    pub entry: bool,
    pub noreturn: bool,
    pub return_param: Option<FunctionParam>,
    pub params: Vec<FunctionParam>,
    pub body: Box<Statement>,
}

#[derive(Debug)]
pub struct FunctionParam {
    pub ident: Ident,
    pub ty: Type,
    pub alignment: Option<u32>,
    pub array_bounds: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateSpace {
    Global,
    Local,
    Shared,
    Register,
    Constant,
    Parameter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    B128,
    B64,
    B32,
    B16,
    B8,
    U64,
    U32,
    U16,
    U8,
    S64,
    S32,
    S16,
    S8,
    F64,
    F32,
    F16x2,
    F16,
    Pred,
}

impl Type {
    pub fn size(&self) -> usize {
        use Type::*;
        match self {
            B128 => 16,
            B64 | U64 | S64 | F64 => 8,
            B32 | U32 | S32 | F32 | F16x2 => 4,
            B16 | U16 | S16 | F16 => 2,
            B8 | U8 | S8 => 1,
            Pred => 1,
        }
    }

    pub fn alignment(&self) -> usize {
        self.size()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Vector {
    V2,
    V4,
}

#[derive(Debug, Clone, Copy)]
pub enum SpecialReg {
    StackPtr,
    ThreadId,
    ThreadIdX,
    ThreadIdY,
    ThreadIdZ,
    NumThread,
    NumThreadX,
    NumThreadY,
    NumThreadZ,
    CtaId,
    CtaIdX,
    CtaIdY,
    CtaIdZ,
    NumCta,
    NumCtaX,
    NumCtaY,
    NumCtaZ,
}

#[derive(Debug, Clone)]
pub struct VarDecl {
    pub state_space: StateSpace,
    pub ty: Type,
    pub vector: Option<Vector>,
    pub ident: Ident,
    pub alignment: Option<u32>,
    pub array_bounds: Vec<u32>,
    pub multiplicity: Option<u32>,
}

#[derive(Debug, Clone)]
pub enum AddressOperand {
    Address(Ident),
    AddressOffset(Ident, i64),
    AddressOffsetVar(Ident, Ident),
    ArrayIndex(Ident, usize),
}

impl AddressOperand {
    pub fn get_ident(&self) -> &Ident {
        match self {
            AddressOperand::Address(ident) => ident,
            AddressOperand::AddressOffset(ident, _) => ident,
            AddressOperand::AddressOffsetVar(ident, _) => ident,
            AddressOperand::ArrayIndex(ident, _) => ident,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Operand {
    SpecialReg(SpecialReg),
    Variable(Ident),
    Immediate(Immediate),
    Address(AddressOperand),
}

#[derive(Debug, Clone, Copy)]
pub enum Immediate {
    Float32(f32),
    Float64(f64),
    Int64(i64),
    UInt64(u64),
}

#[derive(Debug, Clone)]
pub enum Guard {
    Normal(Ident),
    Negated(Ident),
}

#[derive(Debug)]
pub enum Directive {
    VarDecl(VarDecl),
    Version(Version),
    Target(String),
    AddressSize(AddressSize),
    Function(Function),
    Pragma(Pragma),
}

#[derive(Debug, Clone)]
pub struct Instruction {
    pub guard: Option<Guard>,
    pub specifier: Operation,
    pub operands: Vec<Operand>,
}

#[derive(Debug)]
pub enum Statement {
    Directive(Directive),
    Instruction(Instruction),
    Grouping(Vec<Statement>),
    Label(Ident),
}

#[derive(Debug, Clone, Copy)]
pub enum PredicateOp {
    LessThan,
    LessThanEqual,
    GreaterThan,
    GreaterThanEqual,
    Equal,
    NotEqual,
}

#[derive(Debug, Clone, Copy)]
pub enum MulMode {
    Low,
    High,
    Wide,
}

#[derive(Debug, Clone, Copy)]
pub enum RoundingMode {
    NearestEvent,
    Zero,
    NegInf,
    PosInf,
}

#[derive(Debug, Clone)]
pub enum Operation {
    Load(StateSpace, Type),
    Store(StateSpace, Type),
    Move(Type),
    Add(Type),
    Sub(Type),
    Or(Type),
    And(Type),
    FusedMulAdd(RoundingMode, Type),
    Negate(Type),
    Multiply(MulMode, Type),
    MultiplyAdd(MulMode, Type),
    Convert {
        from: Type,
        to: Type,
    },
    ConvertAddress(Type, StateSpace),
    ConvertAddressTo(Type, StateSpace),
    SetPredicate(PredicateOp, Type),
    ShiftLeft(Type),
    Call {
        uniform: bool,
        ident: Ident,
        ret_param: Option<Ident>,
        params: Vec<Ident>,
    },
    BarrierSync,
    Branch,
    Return,
}

pub fn parse_program(src: &str) -> Result<Module, ParseErr> {
    // let res = Token::lexer(src)
    //     .spanned()
    //     .map(|(t, span)| t.map_err(|e| (e, span)))
    //     // .collect::<Result<Vec<_>, _>>();
    //     .collect::<Vec<_>>();
    // dbg!(res);
    let tokens = Token::lexer(src).collect::<Result<Vec<_>, _>>()?;
    let scanner = Scanner::new(&tokens);
    parse_module(scanner).map(|(module, _)| module)
}

fn parse_pragma(mut scanner: Scanner) -> ParseResult<Pragma> {
    scanner.consume(Token::Pragma)?;
    match scanner.must_pop()? {
        Token::StringLiteral(s) => {
            scanner.consume(Token::Semicolon)?;
            Ok((Pragma { value: s.clone() }, scanner))
        }
        t => Err(ParseErr::UnexpectedToken(t.clone())),
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
    let Token::IntegerConst(size) = size else {
        return Err(ParseErr::UnexpectedToken(size.clone()));
    };
    match size {
        32 => Ok((AddressSize::Adr32, scanner)),
        64 => Ok((AddressSize::Adr64, scanner)),
        _ => Ok((AddressSize::Other, scanner)),
    }
}

fn parse_module(mut scanner: Scanner) -> ParseResult<Module> {
    let mut directives = Vec::new();
    while scanner.get().is_some() {
        match parse_directive(scanner) {
            Ok((directive, rest)) => {
                directives.push(directive);
                scanner = rest;
            }
            Err(e) => return Err(e),
        }
    }
    Ok((Module(directives), scanner))
}

fn parse_array_bounds(mut scanner: Scanner) -> ParseResult<Vec<u32>> {
    let mut bounds = Vec::new();
    loop {
        match scanner.get() {
            Some(Token::LeftBracket) => scanner.skip(),
            _ => break Ok((bounds, scanner)),
        }
        let Token::IntegerConst(bound) = scanner.must_pop()? else {
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
        Token::IntegerConst(i) => *i as u32,
        t => return Err(ParseErr::UnexpectedToken(t.clone())),
    };
    Ok((alignment, scanner))
}

fn parse_type(mut scanner: Scanner) -> ParseResult<Type> {
    let ty = match scanner.must_pop()? {
        Token::Bit8 => Type::B8,
        Token::Bit16 => Type::B16,
        Token::Bit32 => Type::B32,
        Token::Bit64 => Type::B64,
        Token::Bit128 => Type::B128,
        Token::Unsigned8 => Type::U8,
        Token::Unsigned16 => Type::U16,
        Token::Unsigned32 => Type::U32,
        Token::Unsigned64 => Type::U64,
        Token::Signed8 => Type::S8,
        Token::Signed16 => Type::S16,
        Token::Signed32 => Type::S32,
        Token::Signed64 => Type::S64,
        Token::Float16 => Type::F16,
        Token::Float16x2 => Type::F16x2,
        Token::Float32 => Type::F32,
        Token::Float64 => Type::F64,
        Token::Predicate => Type::Pred,
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
        Token::NumThreads => SpecialReg::NumThread,
        Token::NumThreadsX => SpecialReg::NumThreadX,
        Token::NumThreadsY => SpecialReg::NumThreadY,
        Token::NumThreadsZ => SpecialReg::NumThreadZ,
        Token::CtaId => SpecialReg::CtaId,
        Token::CtaIdX => SpecialReg::CtaIdX,
        Token::CtaIdY => SpecialReg::CtaIdY,
        Token::CtaIdZ => SpecialReg::CtaIdZ,
        t => return Err(ParseErr::UnexpectedToken(t.clone())),
    };
    Ok((reg, scanner))
}

fn parse_rounding_mode(mut scanner: Scanner) -> ParseResult<RoundingMode> {
    let mode = match scanner.must_pop()? {
        Token::Rn => RoundingMode::NearestEvent,
        Token::Rz => RoundingMode::Zero,
        Token::Rm => RoundingMode::NegInf,
        Token::Rp => RoundingMode::PosInf,
        t => return Err(ParseErr::UnexpectedToken(t.clone())),
    };
    Ok((mode, scanner))
}

fn parse_mul_mode(mut scanner: Scanner) -> ParseResult<MulMode> {
    let mode = match scanner.must_pop()? {
        Token::Low => MulMode::Low,
        Token::High => MulMode::High,
        Token::Wide => MulMode::Wide,
        t => return Err(ParseErr::UnexpectedToken(t.clone())),
    };
    Ok((mode, scanner))
}

fn parse_variable(scanner: Scanner) -> ParseResult<VarDecl> {
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
        VarDecl {
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

fn parse_predicate(mut scanner: Scanner) -> ParseResult<PredicateOp> {
    let pred = match scanner.must_pop()? {
        Token::Ge => PredicateOp::GreaterThanEqual,
        Token::Gt => PredicateOp::GreaterThan,
        Token::Le => PredicateOp::LessThanEqual,
        Token::Lt => PredicateOp::LessThan,
        Token::Eq => PredicateOp::Equal,
        Token::Ne => PredicateOp::NotEqual,
        t => return Err(ParseErr::UnexpectedToken(t.clone())),
    };
    Ok((pred, scanner))
}

fn parse_operation(mut scanner: Scanner) -> ParseResult<Operation> {
    let t = scanner.must_pop()?;
    match t {
        Token::Load => {
            let (state_space, scanner) = parse_state_space(scanner)?;
            let (ty, scanner) = parse_type(scanner)?;
            Ok((Operation::Load(state_space, ty), scanner))
        }
        Token::Store => {
            let (state_space, scanner) = parse_state_space(scanner)?;
            let (ty, scanner) = parse_type(scanner)?;
            Ok((Operation::Store(state_space, ty), scanner))
        }
        Token::Move => {
            let (ty, scanner) = parse_type(scanner)?;
            Ok((Operation::Move(ty), scanner))
        }
        Token::Add => {
            let (ty, scanner) = parse_type(scanner)?;
            Ok((Operation::Add(ty), scanner))
        }
        Token::Sub => {
            let (ty, scanner) = parse_type(scanner)?;
            Ok((Operation::Sub(ty), scanner))
        }
        Token::Or => {
            let (ty, scanner) = parse_type(scanner)?;
            Ok((Operation::Or(ty), scanner))
        }
        Token::And => {
            let (ty, scanner) = parse_type(scanner)?;
            Ok((Operation::And(ty), scanner))
        }
        Token::Mul => {
            let (mode, scanner) = parse_mul_mode(scanner)?;
            let (ty, scanner) = parse_type(scanner)?;
            Ok((Operation::Multiply(mode, ty), scanner))
        }
        Token::MultiplyAdd => {
            let (mode, scanner) = parse_mul_mode(scanner)?;
            let (ty, scanner) = parse_type(scanner)?;
            Ok((Operation::MultiplyAdd(mode, ty), scanner))
        }
        Token::FusedMulAdd => {
            let (mode, scanner) = parse_rounding_mode(scanner)?;
            let (ty, scanner) = parse_type(scanner)?;
            Ok((Operation::FusedMulAdd(mode, ty), scanner))
        }
        Token::Negate => {
            let (ty, scanner) = parse_type(scanner)?;
            Ok((Operation::Negate(ty), scanner))
        }
        Token::Convert => {
            let (to, scanner) = parse_type(scanner)?;
            let (from, scanner) = parse_type(scanner)?;
            Ok((Operation::Convert { to, from }, scanner))
        }
        Token::Call => {
            let uniform = scanner.consume_match(Token::Uniform);
            let ret_param = if let Token::LeftParen = scanner.must_get()? {
                scanner.skip();
                let ident = match scanner.must_pop()? {
                    Token::Identifier(s) => s.clone(),
                    t => return Err(ParseErr::UnexpectedToken(t.clone())),
                };
                scanner.consume(Token::RightParen)?;
                scanner.consume(Token::Comma)?;
                Some(ident)
            } else {
                None
            };
            let ident = match scanner.must_pop()? {
                Token::Identifier(s) => s.clone(),
                t => return Err(ParseErr::UnexpectedToken(t.clone())),
            };
            scanner.consume(Token::Comma)?;
            let mut params = Vec::new();
            if let Token::LeftParen = scanner.must_get()? {
                scanner.skip();
                loop {
                    let ident = match scanner.must_pop()? {
                        Token::Identifier(s) => s.clone(),
                        t => return Err(ParseErr::UnexpectedToken(t.clone())),
                    };
                    params.push(ident);
                    match scanner.must_pop()? {
                        Token::RightParen => break,
                        Token::Comma => {}
                        t => return Err(ParseErr::UnexpectedToken(t.clone())),
                    }
                }
            };

            Ok((
                Operation::Call {
                    uniform,
                    ident,
                    ret_param,
                    params,
                },
                scanner,
            ))
        }
        Token::ConvertAddress => match scanner.get() {
            Some(Token::To) => {
                scanner.skip();
                let (state_space, scanner) = parse_state_space(scanner)?;
                let (ty, scanner) = parse_type(scanner)?;
                Ok((
                    Operation::ConvertAddressTo(ty, state_space),
                    scanner,
                ))
            }
            _ => {
                let (state_space, scanner) = parse_state_space(scanner)?;
                let (ty, scanner) = parse_type(scanner)?;
                Ok((
                    Operation::ConvertAddress(ty, state_space),
                    scanner,
                ))
            }
        },
        Token::SetPredicate => {
            let (pred, scanner) = parse_predicate(scanner)?;
            let (ty, scanner) = parse_type(scanner)?;
            Ok((Operation::SetPredicate(pred, ty), scanner))
        }
        Token::ShiftLeft => {
            let (ty, scanner) = parse_type(scanner)?;
            Ok((Operation::ShiftLeft(ty), scanner))
        }
        Token::Branch => Ok((Operation::Branch, scanner)),
        Token::Return => Ok((Operation::Return, scanner)),
        Token::Bar => {
            // cta token is meaningless
            if let Some(Token::Cta) = scanner.get() {
                scanner.skip();
            }
            scanner.consume(Token::Sync)?;
            Ok((Operation::BarrierSync, scanner))
        }
        t => Err(ParseErr::UnexpectedToken(t.clone())),
    }
}

fn parse_operand(mut scanner: Scanner) -> ParseResult<Operand> {
    // first try to parse a special register
    if let Ok((operand, rest)) = parse_special_reg(scanner) {
        return Ok((Operand::SpecialReg(operand), rest));
    }
    // then fall back to other options
    let operand = match scanner.must_pop()? {
        Token::IntegerConst(i) => Operand::Immediate(Immediate::Int64(*i)),
        Token::Float64Const(f) => Operand::Immediate(Immediate::Float64(*f)),
        Token::Float32Const(f) => Operand::Immediate(Immediate::Float32(*f)),
        Token::Identifier(s) => {
            if let Some(Token::LeftBracket) = scanner.get() {
                // array syntax
                todo!()
            } else {
                Operand::Variable(s.clone())
            }
        }
        Token::LeftBracket => {
            let ident = match scanner.must_pop()? {
                Token::Identifier(s) => s.clone(),
                t => return Err(ParseErr::UnexpectedToken(t.clone())),
            };
            let res = if let Some(Token::Plus) = scanner.get() {
                scanner.skip();
                match scanner.must_pop()? {
                    Token::IntegerConst(i) => {
                        Operand::Address(AddressOperand::AddressOffset(ident, *i))
                    }
                    Token::Identifier(s) => {
                        Operand::Address(AddressOperand::AddressOffsetVar(ident, s.clone()))
                    }
                    t => return Err(ParseErr::UnexpectedToken(t.clone())),
                }
            } else {
                Operand::Address(AddressOperand::Address(ident))
            };
            scanner.consume(Token::RightBracket)?;
            res
        }
        t => return Err(ParseErr::UnexpectedToken(t.clone())),
    };
    Ok((operand, scanner))
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
        let (op, remaining) = parse_operand(scanner)?;
        scanner = remaining;
        operands.push(op);
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
    let (res, scanner) = match scanner.must_get()? {
        Token::Version => {
            let (version, scanner) = parse_version(scanner)?;
            (Directive::Version(version), scanner)
        }
        Token::Target => {
            let (target, scanner) = parse_target(scanner)?;
            (Directive::Target(target), scanner)
        }
        Token::AddressSize => {
            let (addr_size, scanner) = parse_address_size(scanner)?;
            (Directive::AddressSize(addr_size), scanner)
        }
        Token::Function | Token::Visible | Token::Entry => {
            let (function, scanner) = parse_function(scanner)?;
            (Directive::Function(function), scanner)
        }
        Token::Pragma => {
            let (pragma, scanner) = parse_pragma(scanner)?;
            (Directive::Pragma(pragma), scanner)
        }
        _ => {
            let (var, scanner) = parse_variable(scanner)?;
            (Directive::VarDecl(var), scanner)
        }
    };
    Ok((res, scanner))
}

fn parse_instruction(mut scanner: Scanner) -> ParseResult<Instruction> {
    let guard = if let Ok((guard, res)) = parse_guard(scanner) {
        scanner = res;
        Some(guard)
    } else {
        None
    };

    let (specifier, scanner) = parse_operation(scanner)?;
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

fn parse_statement(mut scanner: Scanner) -> ParseResult<Statement> {
    match scanner.must_get()? {
        Token::LeftBrace => {
            let (grouping, scanner) = parse_grouping(scanner)?;
            Ok((Statement::Grouping(grouping), scanner))
        }
        t if t.is_directive() => {
            let (dir, scanner) = parse_directive(scanner)?;
            Ok((Statement::Directive(dir), scanner))
        }
        Token::Identifier(i) => {
            let i = i.clone();
            scanner.skip();
            scanner.consume(Token::Colon)?;
            Ok((Statement::Label(i), scanner))
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
    // if there is no left parenthesis, there are no parameters
    if let Some(Token::LeftParen) = scanner.get() {
        scanner.skip();
    } else {
        return Ok((Vec::new(), scanner));
    }
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

fn parse_return_param(mut scanner: Scanner) -> ParseResult<Option<FunctionParam>> {
    if let Some(Token::LeftParen) = scanner.get() {
        scanner.skip();
    } else {
        return Ok((None, scanner));
    }
    let (param, mut scanner) = parse_function_param(scanner)?;
    scanner.consume(Token::RightParen)?;
    Ok((Some(param), scanner))
}

fn parse_function(mut scanner: Scanner) -> ParseResult<Function> {
    let visible = if let Some(Token::Visible) = scanner.get() {
        scanner.skip();
        true
    } else {
        false
    };
    let entry = match scanner.must_get()? {
        Token::Entry => true,
        Token::Function => false,
        t => return Err(ParseErr::UnexpectedToken(t.clone())),
    };
    scanner.skip();

    let (return_param, mut scanner) = parse_return_param(scanner)?;

    let ident = match scanner.must_pop()? {
        Token::Identifier(s) => s.clone(),
        t => return Err(ParseErr::UnexpectedToken(t.clone())),
    };
    let noreturn = if let Some(Token::NoReturn) = scanner.get() {
        scanner.skip();
        true
    } else {
        false
    };

    let (params, scanner) = parse_function_params(scanner)?;
    let (body, scanner) = parse_statement(scanner)?;

    Ok((
        Function {
            ident,
            visible,
            entry,
            return_param,
            noreturn,
            params,
            body: Box::new(body),
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
        let contents = std::fs::read_to_string("kernels/add.ptx").unwrap();
        let _ = parse_program(&contents).unwrap();
    }

    #[test]
    fn test_parse_transpose() {
        let contents = std::fs::read_to_string("kernels/transpose.ptx").unwrap();
        let _ = parse_program(&contents).unwrap();
    }

    #[test]
    fn test_parse_add_simple() {
        let contents = std::fs::read_to_string("kernels/add_simple.ptx").unwrap();
        let _ = parse_program(&contents).unwrap();
    }

    #[test]
    fn test_parse_test() {
        let contents = std::fs::read_to_string("kernels/test.ptx").unwrap();
        let _ = parse_program(&contents).unwrap();
    }

    #[test]
    fn test_parse_gemm() {
        let contents = std::fs::read_to_string("kernels/gemm.ptx").unwrap();
        let _ = parse_program(&contents).unwrap();
    }
}
