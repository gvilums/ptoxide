use std::ops::Range;

use logos::{Lexer, Logos};
use thiserror::Error;

fn lex_reg_multiplicity<'a>(lex: &mut Lexer<'a, Token<'a>>) -> Result<u32, LexError> {
    let mut s = lex.slice();
    s = &s[1..s.len() - 1];
    s.parse().map_err(|_| LexError::ParseRegMultiplicity)
}

fn lex_version_number<'a>(lex: &mut Lexer<'a, Token<'a>>) -> Result<Version, LexError> {
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

fn lex_float32_constant<'a>(lex: &mut Lexer<'a, Token<'a>>) -> Result<f32, LexError> {
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

fn lex_float64_constant<'a>(_lex: &mut Lexer<'a, Token<'a>>) -> Option<f64> {
    todo!()
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum LexError {
    ParseFloatConst,
    ParseRegMultiplicity,
    ParseVersionNumber,
    #[default]
    Unknown,
}

impl std::fmt::Display for LexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
impl std::error::Error for LexError {}

#[derive(Logos, Debug, PartialEq, Clone, Copy)]
#[logos(skip r"[ \t\n\f]+")] // Ignore this regex pattern between tokens
#[logos(error = LexError)]
pub enum Token<'a> {
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
    #[token("not")]
    Not,
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

    #[regex(r"[a-zA-Z][a-zA-Z0-9_$]*|[_$%][a-zA-Z0-9_$]+", |lex| lex.slice())]
    Identifier(&'a str),

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

    #[regex(r#""[^"]*""#, |lex| lex.slice())]
    StringLiteral(&'a str),
    #[regex(r"\d+\.\d+", lex_version_number)]
    VersionNumber(Version),

    #[regex(r"//.*", logos::skip)]
    Skip,
}

impl<'a> Token<'a> {
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

impl<'a> std::fmt::Display for Token<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SourceLocation {
    byte: usize,
    line: usize,
    col: usize,
}

impl std::fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "@{}:{} (byte {})", self.line, self.col, self.byte)
    }
}

#[derive(Error, Debug)]
pub enum ParseErr {
    #[error("Unexpected token \"{:?}\"", .0)]
    UnexpectedToken(String, SourceLocation),
    #[error("Unexpected end of file")]
    UnexpectedEof,
    #[error("Lex error \"{:?}\" at {:?}", .0, .1)]
    LexError(LexError, SourceLocation),
    #[error("Unknown token \"{:?}\" at {:?}", .0, .1)]
    UnknownToken(String, SourceLocation),
}

type ParseResult<'a, T> = Result<T, ParseErr>;

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

impl From<SpecialReg> for Operand {
    fn from(value: SpecialReg) -> Self {
        Operand::SpecialReg(value)
    }
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
    Not(Type),
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

type TokenPos<'a> = Range<usize>;

struct Parser<'a> {
    src: &'a str,
    inner: std::iter::Peekable<logos::SpannedIter<'a, Token<'a>>>,
}

impl<'a> Parser<'a> {
    pub fn new(src: &'a str) -> Self {
        Self {
            src,
            inner: Token::lexer(src).spanned().peekable(),
        }
    }

    fn locate(&self, span: Range<usize>) -> SourceLocation {
        let text = self.src.as_bytes();

        let mut line = 1;
        let mut col = 0;

        let end = span.end.min(text.len());

        for &c in &text[..end] {
            if c == b'\n' {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }
        }

        SourceLocation {
            byte: span.start,
            line,
            col,
        }
    }

    fn unexpected(&self, (token, pos): (Token, TokenPos)) -> ParseErr {
        ParseErr::UnexpectedToken(token.to_string(), self.locate(pos))
    }

    fn get(&mut self) -> ParseResult<Option<(Token<'a>, TokenPos)>> {
        match self.inner.peek().cloned() {
            Some((Ok(tok), pos)) => Ok(Some((tok, pos))),
            Some((Err(LexError::Unknown), pos)) => Err(ParseErr::UnknownToken(
                self.src[pos.clone()].to_string(),
                self.locate(pos),
            )),
            Some((Err(err), pos)) => Err(ParseErr::LexError(err, self.locate(pos))),
            None => Ok(None),
        }
    }

    fn must_get(&mut self) -> Result<(Token<'a>, TokenPos), ParseErr> {
        self.get()?.ok_or(ParseErr::UnexpectedEof)
    }

    fn skip(&mut self) {
        self.inner.next();
    }

    fn consume(&mut self, token: Token) -> ParseResult<()> {
        let head = self.must_get()?;
        if head.0 == token {
            self.skip();
            Ok(())
        } else {
            Err(self.unexpected(head))
        }
    }

    fn consume_match(&mut self, token: Token) -> ParseResult<bool> {
        let Some(head) = self.get()? else {
            return Ok(false);
        };
        if head.0 == token {
            self.skip();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn pop(&mut self) -> ParseResult<Option<(Token<'a>, TokenPos)>> {
        match self.inner.next() {
            Some((Ok(tok), pos)) => Ok(Some((tok, pos))),
            Some((Err(err), pos)) => Err(ParseErr::LexError(err, self.locate(pos))),
            None => Ok(None),
        }
    }

    fn must_pop(&mut self) -> Result<(Token<'a>, TokenPos), ParseErr> {
        self.pop()?.ok_or(ParseErr::UnexpectedEof)
    }

    fn parse_pragma(&mut self) -> ParseResult<Pragma> {
        self.consume(Token::Pragma)?;
        let t = self.must_pop()?;
        match t.0 {
            Token::StringLiteral(s) => {
                self.consume(Token::Semicolon)?;
                Ok(Pragma {
                    value: s.to_string(),
                })
            }
            _ => Err(self.unexpected(t)),
        }
    }

    fn parse_version(&mut self) -> ParseResult<Version> {
        self.consume(Token::Version)?;
        let t = self.must_pop()?;
        match t.0 {
            Token::VersionNumber(version) => Ok(version),
            _ => Err(self.unexpected(t)),
        }
    }

    fn parse_target(&mut self) -> ParseResult<String> {
        self.consume(Token::Target)?;
        let t = self.must_pop()?;
        match t.0 {
            Token::Identifier(target) => Ok(target.to_string()),
            _ => Err(self.unexpected(t)),
        }
    }

    fn parse_address_size(&mut self) -> ParseResult<AddressSize> {
        self.consume(Token::AddressSize)?;
        let t = self.must_pop()?;
        let Token::IntegerConst(size) = t.0 else {
            return Err(self.unexpected(t));
        };
        match size {
            32 => Ok(AddressSize::Adr32),
            64 => Ok(AddressSize::Adr64),
            _ => Ok(AddressSize::Other),
        }
    }

    fn parse_module(&mut self) -> ParseResult<Module> {
        let mut directives = Vec::new();
        while self.get()?.is_some() {
            match self.parse_directive() {
                Ok(directive) => {
                    directives.push(directive);
                }
                Err(e) => return Err(e),
            }
        }
        Ok(Module(directives))
    }

    fn parse_array_bounds(&mut self) -> ParseResult<Vec<u32>> {
        let mut bounds = Vec::new();
        loop {
            match self.get()? {
                Some((Token::LeftBracket, _)) => self.skip(),
                _ => break Ok(bounds),
            }
            let t = self.must_pop()?;
            let Token::IntegerConst(bound) = t.0 else {
                return Err(self.unexpected(t));
            };
            self.consume(Token::RightBracket)?;
            // todo clean up raw casts
            bounds.push(bound as u32);
        }
    }

    fn parse_state_space(&mut self) -> ParseResult<StateSpace> {
        let t = self.must_pop()?;
        match t.0 {
            Token::Global => Ok(StateSpace::Global),
            Token::Local => Ok(StateSpace::Local),
            Token::Shared => Ok(StateSpace::Shared),
            Token::Reg => Ok(StateSpace::Register),
            Token::Param => Ok(StateSpace::Parameter),
            Token::Const => Ok(StateSpace::Constant),
            _ => Err(self.unexpected(t)),
        }
    }

    fn parse_alignment(&mut self) -> ParseResult<u32> {
        self.consume(Token::Align)?;
        let t = self.must_pop()?;
        let alignment = match t.0 {
            Token::IntegerConst(i) => i as u32,
            _ => return Err(self.unexpected(t)),
        };
        Ok(alignment)
    }

    fn parse_type(&mut self) -> ParseResult<Type> {
        let t = self.must_pop()?;
        let ty = match t.0 {
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
            _ => return Err(self.unexpected(t)),
        };
        Ok(ty)
    }

    fn parse_rounding_mode(&mut self) -> ParseResult<RoundingMode> {
        let t = self.must_pop()?;
        let mode = match t.0 {
            Token::Rn => RoundingMode::NearestEvent,
            Token::Rz => RoundingMode::Zero,
            Token::Rm => RoundingMode::NegInf,
            Token::Rp => RoundingMode::PosInf,
            _ => return Err(self.unexpected(t)),
        };
        Ok(mode)
    }

    fn parse_mul_mode(&mut self) -> ParseResult<MulMode> {
        let t = self.must_pop()?;
        let mode = match t.0 {
            Token::Low => MulMode::Low,
            Token::High => MulMode::High,
            Token::Wide => MulMode::Wide,
            _ => return Err(self.unexpected(t)),
        };
        Ok(mode)
    }

    fn parse_variable(&mut self) -> ParseResult<VarDecl> {
        let state_space = self.parse_state_space()?;

        let t = self.get()?;
        let alignment = if let Some((Token::Align, _)) = t {
            Some(self.parse_alignment()?)
        } else {
            None
        };

        let t = self.get()?;
        let vector = match t {
            Some((Token::V2, _)) => {
                self.skip();
                Some(Vector::V2)
            }
            Some((Token::V4, _)) => {
                self.skip();
                Some(Vector::V4)
            }
            _ => None,
        };

        let ty = self.parse_type()?;

        let t = self.must_pop()?;
        let ident = match t.0 {
            Token::Identifier(s) => s.to_string(),
            _ => return Err(self.unexpected(t)),
        };

        let t = self.must_get()?;
        let multiplicity = match t.0 {
            Token::RegMultiplicity(m) => {
                self.skip();
                Some(m)
            }
            _ => None,
        };

        let array_bounds = self.parse_array_bounds()?;

        self.consume(Token::Semicolon)?;

        Ok(VarDecl {
            state_space,
            ty,
            vector,
            alignment,
            array_bounds,
            ident: ident.to_string(),
            multiplicity,
        })
    }

    fn parse_guard(&mut self) -> ParseResult<Guard> {
        self.consume(Token::At)?;
        let t = self.must_pop()?;
        let guard = match t.0 {
            Token::Identifier(s) => Guard::Normal(s.to_string()),
            Token::Exclamation => {
                let t = self.must_pop()?;
                let ident = match t.0 {
                    Token::Identifier(s) => s,
                    _ => return Err(self.unexpected(t)),
                };
                Guard::Negated(ident.to_string())
            }
            _ => return Err(self.unexpected(t)),
        };
        Ok(guard)
    }

    fn parse_predicate(&mut self) -> ParseResult<PredicateOp> {
        let t = self.must_pop()?;
        let pred = match t.0 {
            Token::Ge => PredicateOp::GreaterThanEqual,
            Token::Gt => PredicateOp::GreaterThan,
            Token::Le => PredicateOp::LessThanEqual,
            Token::Lt => PredicateOp::LessThan,
            Token::Eq => PredicateOp::Equal,
            Token::Ne => PredicateOp::NotEqual,
            _ => return Err(self.unexpected(t)),
        };
        Ok(pred)
    }

    fn parse_operation(&mut self) -> ParseResult<Operation> {
        let t = self.must_pop()?;
        match t.0 {
            Token::Load => {
                let state_space = self.parse_state_space()?;
                let ty = self.parse_type()?;
                Ok(Operation::Load(state_space, ty))
            }
            Token::Store => {
                let state_space = self.parse_state_space()?;
                let ty = self.parse_type()?;
                Ok(Operation::Store(state_space, ty))
            }
            Token::Move => {
                let ty = self.parse_type()?;
                Ok(Operation::Move(ty))
            }
            Token::Add => {
                let ty = self.parse_type()?;
                Ok(Operation::Add(ty))
            }
            Token::Sub => {
                let ty = self.parse_type()?;
                Ok(Operation::Sub(ty))
            }
            Token::Or => {
                let ty = self.parse_type()?;
                Ok(Operation::Or(ty))
            }
            Token::And => {
                let ty = self.parse_type()?;
                Ok(Operation::And(ty))
            }
            Token::Not => {
                let ty = self.parse_type()?;
                Ok(Operation::Not(ty))
            }
            Token::Mul => {
                let mode = self.parse_mul_mode()?;
                let ty = self.parse_type()?;
                Ok(Operation::Multiply(mode, ty))
            }
            Token::MultiplyAdd => {
                let mode = self.parse_mul_mode()?;
                let ty = self.parse_type()?;
                Ok(Operation::MultiplyAdd(mode, ty))
            }
            Token::FusedMulAdd => {
                let mode = self.parse_rounding_mode()?;
                let ty = self.parse_type()?;
                Ok(Operation::FusedMulAdd(mode, ty))
            }
            Token::Negate => {
                let ty = self.parse_type()?;
                Ok(Operation::Negate(ty))
            }
            Token::Convert => {
                let to = self.parse_type()?;
                let from = self.parse_type()?;
                Ok(Operation::Convert { to, from })
            }
            Token::Call => {
                let uniform = self.consume_match(Token::Uniform)?;
                let ret_param = if let Token::LeftParen = self.must_get()?.0 {
                    self.skip();
                    let t = self.must_pop()?;
                    let ident = match t.0 {
                        Token::Identifier(s) => s.to_string(),
                        _ => return Err(self.unexpected(t)),
                    };
                    self.consume(Token::RightParen)?;
                    self.consume(Token::Comma)?;
                    Some(ident)
                } else {
                    None
                };
                let t = self.must_pop()?;
                let ident = match t.0 {
                    Token::Identifier(s) => s.to_string(),
                    _ => return Err(self.unexpected(t)),
                };
                self.consume(Token::Comma)?;
                let mut params = Vec::new();
                if let Token::LeftParen = self.must_get()?.0 {
                    self.skip();
                    loop {
                        let t = self.must_pop()?;
                        let ident = match t.0 {
                            Token::Identifier(s) => s.to_string(),
                            _ => return Err(self.unexpected(t)),
                        };
                        params.push(ident);
                        let t = self.must_pop()?;
                        match t.0 {
                            Token::RightParen => break,
                            Token::Comma => {}
                            _ => return Err(self.unexpected(t)),
                        }
                    }
                };

                Ok(Operation::Call {
                    uniform,
                    ident: ident.to_string(),
                    ret_param,
                    params,
                })
            }
            Token::ConvertAddress => match self.must_get()?.0 {
                Token::To => {
                    self.skip();
                    let state_space = self.parse_state_space()?;
                    let ty = self.parse_type()?;
                    Ok(Operation::ConvertAddressTo(ty, state_space))
                }
                _ => {
                    let state_space = self.parse_state_space()?;
                    let ty = self.parse_type()?;
                    Ok(Operation::ConvertAddress(ty, state_space))
                }
            },
            Token::SetPredicate => {
                let pred = self.parse_predicate()?;
                let ty = self.parse_type()?;
                Ok(Operation::SetPredicate(pred, ty))
            }
            Token::ShiftLeft => {
                let ty = self.parse_type()?;
                Ok(Operation::ShiftLeft(ty))
            }
            Token::Branch => {
                self.consume_match(Token::Uniform)?;
                Ok(Operation::Branch)
            }
            Token::Return => Ok(Operation::Return),
            Token::Bar => {
                // cta token is meaningless
                self.consume_match(Token::Cta)?;
                self.consume(Token::Sync)?;
                Ok(Operation::BarrierSync)
            }
            _ => Err(self.unexpected(t)),
        }
    }

    fn parse_operand(&mut self) -> ParseResult<Operand> {
        let t = self.must_pop()?;
        let operand = match t.0 {
            Token::ThreadId => SpecialReg::ThreadId.into(),
            Token::ThreadIdX => SpecialReg::ThreadIdX.into(),
            Token::ThreadIdY => SpecialReg::ThreadIdY.into(),
            Token::ThreadIdZ => SpecialReg::ThreadIdZ.into(),
            Token::NumThreads => SpecialReg::NumThread.into(),
            Token::NumThreadsX => SpecialReg::NumThreadX.into(),
            Token::NumThreadsY => SpecialReg::NumThreadY.into(),
            Token::NumThreadsZ => SpecialReg::NumThreadZ.into(),
            Token::CtaId => SpecialReg::CtaId.into(),
            Token::CtaIdX => SpecialReg::CtaIdX.into(),
            Token::CtaIdY => SpecialReg::CtaIdY.into(),
            Token::CtaIdZ => SpecialReg::CtaIdZ.into(),
            Token::IntegerConst(i) => Operand::Immediate(Immediate::Int64(i)),
            Token::Float64Const(f) => Operand::Immediate(Immediate::Float64(f)),
            Token::Float32Const(f) => Operand::Immediate(Immediate::Float32(f)),
            Token::Identifier(s) => {
                let t = self.get()?;
                if let Some((Token::LeftBracket, _)) = t {
                    todo!("array syntax in operands")
                } else {
                    Operand::Variable(s.to_string())
                }
            }
            Token::LeftBracket => {
                let t = self.must_pop()?;
                let Token::Identifier(s) = t.0 else {
                    return Err(self.unexpected(t));
                };
                let ident = s.to_string();

                let t = self.must_get()?;
                let res = if let Token::Plus = t.0 {
                    self.skip();
                    let t = self.must_pop()?;
                    match t.0 {
                        Token::IntegerConst(i) => {
                            Operand::Address(AddressOperand::AddressOffset(ident, i))
                        }
                        Token::Identifier(s) => {
                            Operand::Address(AddressOperand::AddressOffsetVar(ident, s.to_string()))
                        }
                        _ => return Err(self.unexpected(t)),
                    }
                } else {
                    Operand::Address(AddressOperand::Address(ident))
                };
                self.consume(Token::RightBracket)?;
                res
            }
            _ => return Err(self.unexpected(t)),
        };
        Ok(operand)
    }

    fn parse_operands(&mut self) -> ParseResult<Vec<Operand>> {
        let mut operands = Vec::new();
        loop {
            let t = self.must_get()?;
            match t.0 {
                Token::Semicolon => {
                    self.skip();
                    break Ok(operands);
                }
                Token::Comma => self.skip(),
                _ => {}
            }
            let op = self.parse_operand()?;
            operands.push(op);
        }
    }

    fn parse_grouping(&mut self) -> ParseResult<Vec<Statement>> {
        self.consume(Token::LeftBrace)?; // Consume the left brace
        let mut statements = Vec::new();
        loop {
            let t = self.must_get()?;
            if let Token::RightBrace = t.0 {
                self.skip();
                break Ok(statements);
            }
            statements.push(self.parse_statement()?);
        }
    }

    fn parse_directive(&mut self) -> ParseResult<Directive> {
        let t = self.must_get()?;
        let res = match t.0 {
            Token::Version => {
                let version = self.parse_version()?;
                Directive::Version(version)
            }
            Token::Target => {
                let target = self.parse_target()?;
                Directive::Target(target)
            }
            Token::AddressSize => {
                let addr_size = self.parse_address_size()?;
                Directive::AddressSize(addr_size)
            }
            Token::Function | Token::Visible | Token::Entry => {
                let function = self.parse_function()?;
                Directive::Function(function)
            }
            Token::Pragma => {
                let pragma = self.parse_pragma()?;
                Directive::Pragma(pragma)
            }
            _ => {
                let var = self.parse_variable()?;
                Directive::VarDecl(var)
            }
        };
        Ok(res)
    }

    fn parse_instruction(&mut self) -> ParseResult<Instruction> {
        let t = self.must_get()?;
        let guard = if let Token::At = t.0 {
            Some(self.parse_guard()?)
        } else {
            None
        };

        let specifier = self.parse_operation()?;
        let operands = self.parse_operands()?;

        Ok(Instruction {
            guard,
            specifier,
            operands,
        })
    }

    fn parse_statement(&mut self) -> ParseResult<Statement> {
        let t = self.must_get()?;
        match t.0 {
            Token::LeftBrace => {
                let grouping = self.parse_grouping()?;
                Ok(Statement::Grouping(grouping))
            }
            t if t.is_directive() => {
                let dir = self.parse_directive()?;
                Ok(Statement::Directive(dir))
            }
            Token::Identifier(i) => {
                let i = i.to_string();
                self.skip();
                self.consume(Token::Colon)?;
                Ok(Statement::Label(i.to_string()))
            }
            _ => {
                let instr = self.parse_instruction()?;
                Ok(Statement::Instruction(instr))
            }
        }
    }

    fn parse_function_param(&mut self) -> ParseResult<FunctionParam> {
        self.consume(Token::Param)?; // Consume the param keyword

        let alignment = None; // todo parse alignment in function param

        let ty = self.parse_type()?;
        let ident = loop {
            let t = self.must_pop()?;
            if let Token::Identifier(s) = t.0 {
                break s.to_string();
            }
        };

        let array_bounds = self.parse_array_bounds()?;

        Ok(FunctionParam {
            alignment,
            ident: ident.to_string(),
            ty,
            array_bounds,
        })
    }

    fn parse_function_params(&mut self) -> ParseResult<Vec<FunctionParam>> {
        // if there is no left parenthesis, there are no parameters
        if !self.consume_match(Token::LeftParen)? {
            return Ok(Vec::new());
        }
        // if we immediately see a right parenthesis, there are no parameters
        if self.consume_match(Token::RightParen)? {
            return Ok(Vec::new());
        }

        let mut params = Vec::new();
        loop {
            params.push(self.parse_function_param()?);
            let t = self.must_pop()?;
            match t.0 {
                Token::Comma => {}
                Token::RightParen => break Ok(params),
                _ => return Err(self.unexpected(t)),
            }
        }
    }

    fn parse_return_param(&mut self) -> ParseResult<Option<FunctionParam>> {
        let t = self.must_get()?;
        if let Token::LeftParen = t.0 {
            self.skip();
        } else {
            return Ok(None);
        }
        let param = self.parse_function_param()?;
        self.consume(Token::RightParen)?;
        Ok(Some(param))
    }

    fn parse_function(&mut self) -> ParseResult<Function> {
        let visible = if let Token::Visible = self.must_get()?.0 {
            self.skip();
            true
        } else {
            false
        };
        let t = self.must_pop()?;
        let entry = match t.0 {
            Token::Entry => true,
            Token::Function => false,
            _ => return Err(self.unexpected(t)),
        };

        let return_param = self.parse_return_param()?;

        let t = self.must_pop()?;
        let ident = match t.0 {
            Token::Identifier(s) => s.to_string(),
            _ => return Err(self.unexpected(t)),
        };

        let noreturn = if let Token::NoReturn = self.must_get()?.0 {
            self.skip();
            true
        } else {
            false
        };

        let params = self.parse_function_params()?;
        let body = self.parse_statement()?;

        Ok(Function {
            ident: ident.to_string(),
            visible,
            entry,
            return_param,
            noreturn,
            params,
            body: Box::new(body),
        })
    }
}

pub fn parse_program(src: &str) -> Result<Module, ParseErr> {
    Parser::new(src).parse_module()
}

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
    fn test_parse_fncall() {
        let contents = std::fs::read_to_string("kernels/fncall.ptx").unwrap();
        let _ = parse_program(&contents).unwrap();
    }

    #[test]
    fn test_parse_gemm() {
        let contents = std::fs::read_to_string("kernels/gemm.ptx").unwrap();
        let _ = parse_program(&contents).unwrap();
    }
}
