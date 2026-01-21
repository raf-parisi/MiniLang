#ifndef TOKEN_H
#define TOKEN_H

#include <string>

enum TokenType {
    TOK_EOF = -1,
    TOK_PRINT = -2,
    TOK_NUMBER = -3,
    TOK_IDENTIFIER = -4,
    TOK_PLUS = -5,
    TOK_MINUS = -6,
    TOK_STAR = -7,
    TOK_SLASH = -8,
    TOK_ASSIGN = -9,
    TOK_SEMICOLON = -10,
    TOK_UNKNOWN = -11
};

struct Token {
    TokenType type;
    std::string lexeme;
    double value;
    int line;
    int column;

    Token(TokenType t = TOK_UNKNOWN, std::string lex = "", double val = 0.0, int l = 1, int c = 1)
        : type(t), lexeme(lex), value(val), line(l), column(c) {}
};

#endif
