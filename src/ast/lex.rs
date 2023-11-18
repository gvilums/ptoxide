use logos::{Logos, Lexer};

fn lex_reg_multiplicity<'a>(lex: &mut Lexer<'a, Token<'a>>) -> Result<u32, LexError> {
    let mut s = lex.slice();
    s = &s[1..s.len() - 1];
    s.parse().map_err(|_| LexError::ParseRegMultiplicity)
}

fn lex_version_number<'a>(lex: &mut Lexer<'a, Token<'a>>) -> Result<(u32, u32), LexError> {
    let num_str = lex.slice().split_whitespace().nth(1).ok_or(LexError::ParseVersionNumber)?;
    let Some((major_str, minor_str)) = num_str.split_once('.') else {
        return Err(LexError::ParseVersionNumber);
    };
    let major = major_str
        .parse()
        .map_err(|_| LexError::ParseVersionNumber)?;
    let minor = minor_str
        .parse()
        .map_err(|_| LexError::ParseVersionNumber)?;
    Ok(( major, minor ))
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

fn lex_float64_constant<'a>(lex: &mut Lexer<'a, Token<'a>>) -> Result<f64, LexError> {
    let Some(vals) = lex.slice().as_bytes().get(2..) else {
        return Err(LexError::ParseFloatConst);
    };
    let mut val = 0u64;
    for c in vals {
        val <<= 4;
        val |= match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            _ => return Err(LexError::ParseFloatConst),
        } as u64;
    }
    Ok(f64::from_bits(val))
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
    #[token(".address_size")]
    AddressSize,
    #[token(".explicitcluster")]
    Explicitcluster,
    #[token(".maxnreg")]
    Maxnreg,
    #[token(".section")]
    Section,
    #[token(".alias")]
    Alias,
    #[token(".extern")]
    Extern,
    #[token(".maxntid")]
    Maxntid,
    #[token(".shared")]
    Shared,
    #[token(".align")]
    Align,
    #[token(".file")]
    File,
    #[token(".minnctapersm")]
    Minnctapersm,
    #[token(".sreg")]
    Sreg,
    #[token(".branchtargets")]
    Branchtargets,
    #[token(".func")]
    Func,
    #[token(".noreturn")]
    Noreturn,
    #[token(".target")]
    Target,
    #[token(".callprototype")]
    Callprototype,
    #[token(".global")]
    Global,
    #[token(".param")]
    Param,
    #[token(".tex")]
    Tex,
    #[token(".calltargets")]
    Calltargets,
    #[token(".loc")]
    Loc,
    #[token(".pragma")]
    Pragma,
    // we parse this as a single token to avoid ambiguity with float constants
    #[regex(r".version[ \t\f\n]+\d+\.\d+", lex_version_number)]
    Version((u32, u32)),
    #[token(".common")]
    Common,
    #[token(".local")]
    Local,
    #[token(".reg")]
    Reg,
    #[token(".visible")]
    Visible,
    #[token(".const")]
    Const,
    #[token(".maxclusterrank")]
    Maxclusterrank,
    #[token(".reqnctapercluster")]
    Reqnctapercluster,
    #[token(".weak")]
    Weak,
    #[token(".entry")]
    Entry,
    #[token(".maxnctapersm")]
    Maxnctapersm,
    #[token(".reqntid")]
    Reqntid,

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

    #[token("abs")]
    Abs,
    #[token("discard")]
    Discard,
    #[token("min")]
    Min,
    #[token("shf")]
    Shf,
    #[token("vadd")]
    Vadd,
    #[token("activemask")]
    Activemask,
    #[token("div")]
    Div,
    #[token("mma")]
    Mma,
    #[token("shfl")]
    Shfl,
    #[token("vadd2")]
    Vadd2,
    #[token("add")]
    Add,
    #[token("dp2a")]
    Dp2A,
    #[token("mov")]
    Mov,
    #[token("shl")]
    Shl,
    #[token("vadd4")]
    Vadd4,
    #[token("addc")]
    Addc,
    #[token("dp4a")]
    Dp4A,
    #[token("movmatrix")]
    Movmatrix,
    #[token("shr")]
    Shr,
    #[token("vavrg2")]
    Vavrg2,
    #[token("alloca")]
    Alloca,
    #[token("elect")]
    Elect,
    #[token("mul")]
    Mul,
    #[token("sin")]
    Sin,
    #[token("vavrg4")]
    Vavrg4,
    #[token("and")]
    And,
    #[token("ex2")]
    Ex2,
    #[token("mul24")]
    Mul24,
    #[token("slct")]
    Slct,
    #[token("vmad")]
    Vmad,
    #[token("applypriority")]
    Applypriority,
    #[token("exit")]
    Exit,
    #[token("multimem")]
    Multimem,
    #[token("sqrt")]
    Sqrt,
    #[token("vmax")]
    Vmax,
    #[token("atom")]
    Atom,
    #[token("fence")]
    Fence,
    #[token("nanosleep")]
    Nanosleep,
    #[token("st")]
    St,
    #[token("vmax2")]
    Vmax2,
    #[token("bar")]
    Bar,
    #[token("fma")]
    Fma,
    #[token("neg")]
    Neg,
    #[token("stackrestore")]
    Stackrestore,
    #[token("vmax4")]
    Vmax4,
    #[token("barrier")]
    Barrier,
    #[token("fns")]
    Fns,
    #[token("not")]
    Not,
    #[token("stacksave")]
    Stacksave,
    #[token("vmin")]
    Vmin,
    #[token("bfe")]
    Bfe,
    #[token("getctarank")]
    Getctarank,
    #[token("or")]
    Or,
    #[token("stmatrix")]
    Stmatrix,
    #[token("vmin2")]
    Vmin2,
    #[token("bfi")]
    Bfi,
    #[token("griddepcontrol")]
    Griddepcontrol,
    #[token("pmevent")]
    Pmevent,
    #[token("sub")]
    Sub,
    #[token("vmin4")]
    Vmin4,
    #[token("bfind")]
    Bfind,
    #[token("isspacep")]
    Isspacep,
    #[token("popc")]
    Popc,
    #[token("subc")]
    Subc,
    #[token("vote")]
    Vote,
    #[token("bmsk")]
    Bmsk,
    #[token("istypep")]
    Istypep,
    #[token("prefetch")]
    Prefetch,
    #[token("suld")]
    Suld,
    #[token("vset")]
    Vset,
    #[token("bra")]
    Bra,
    #[token("ld")]
    Ld,
    #[token("prefetchu")]
    Prefetchu,
    #[token("suq")]
    Suq,
    #[token("vset2")]
    Vset2,
    #[token("brev")]
    Brev,
    #[token("ldmatrix")]
    Ldmatrix,
    #[token("prmt")]
    Prmt,
    #[token("sured")]
    Sured,
    #[token("vset4")]
    Vset4,
    #[token("brkpt")]
    Brkpt,
    #[token("ldu")]
    Ldu,
    #[token("rcp")]
    Rcp,
    #[token("sust")]
    Sust,
    #[token("vshl")]
    Vshl,
    #[token("brx")]
    Brx,
    #[token("lg2")]
    Lg2,
    #[token("red")]
    Red,
    #[token("szext")]
    Szext,
    #[token("vshr")]
    Vshr,
    #[token("call")]
    Call,
    #[token("lop3")]
    Lop3,
    #[token("redux")]
    Redux,
    #[token("tanh")]
    Tanh,
    #[token("vsub")]
    Vsub,
    #[token("clz")]
    Clz,
    #[token("mad")]
    Mad,
    #[token("rem")]
    Rem,
    #[token("testp")]
    Testp,
    #[token("vsub2")]
    Vsub2,
    #[token("cnot")]
    Cnot,
    #[token("mad24")]
    Mad24,
    #[token("ret")]
    Ret,
    #[token("tex")]
    InsTex,
    #[token("vsub4")]
    Vsub4,
    #[token("copysign")]
    Copysign,
    #[token("madc")]
    Madc,
    #[token("rsqrt")]
    Rsqrt,
    #[token("tld4")]
    Tld4,
    #[token("wgmma")]
    Wgmma,
    #[token("cos")]
    Cos,
    #[token("mapa")]
    Mapa,
    #[token("sad")]
    Sad,
    #[token("trap")]
    Trap,
    #[token("wmma")]
    Wmma,
    #[token("cp")]
    Cp,
    #[token("match")]
    Match,
    #[token("selp")]
    Selp,
    #[token("txq")]
    Txq,
    #[token("xor")]
    Xor,
    #[token("createpolicy")]
    Createpolicy,
    #[token("max")]
    Max,
    #[token("set")]
    Set,
    #[token("vabsdiff")]
    Vabsdiff,
    #[token("cvt")]
    Cvt,
    #[token("mbarrier")]
    Mbarrier,
    #[token("setmaxnreg")]
    Setmaxnreg,
    #[token("vabsdiff2")]
    Vabsdiff2,
    #[token("cvta")]
    Cvta,
    #[token("membar")]
    Membar,
    #[token("setp")]
    Setp,
    #[token("vabsdiff4")]
    Vabsdiff4,

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
    // #[regex(r"[-+]?[0-9]*\.([0-9]+([eE][-+]?[0-9]+)?)", |lex| lex.slice().parse().ok())]
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
    Bang,
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
    VersionNumber((u32, u32)),

    #[regex(r"//.*", logos::skip)]
    Skip,
}

impl<'a> Token<'a> {
    pub fn is_directive(&self) -> bool {
        matches!(
            self,
            Token::Version(_)
                | Token::Target
                | Token::AddressSize
                | Token::Visible
                | Token::Entry
                | Token::Func
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
