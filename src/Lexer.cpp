#include "Lexer.h"
#include <cctype>

Lexer::Lexer(const std::string& source) : input(source), pos(0), line(1), column(1) {}

char Lexer::peek() {
    if (pos >= input.size()) return '\0';
    return input[pos];
}

char Lexer::advance() {
    if (pos >= input.size()) return '\0';
    char c = input[pos++];
    if (c == '\n') {
        line++;
        column = 1;
    } else {
        column++;
    }
    return c;
}

void Lexer::skipWhitespace() {
    while (std::isspace(peek())) {
        advance();
    }
}

Token Lexer::number() {
    int startCol = column;
    std::string num;
    while (std::isdigit(peek()) || peek() == '.') {
        num += advance();
    }
    return Token(TOK_NUMBER, num, std::stod(num), line, startCol);
}

Token Lexer::identifier() {
    int startCol = column;
    std::string id;
    while (std::isalnum(peek()) || peek() == '_') {
        id += advance();
    }
    
    // Check for keywords
    if (id == "print") {
        return Token(TOK_PRINT, id, 0, line, startCol);
    }
    if (id == "fn") {
        return Token(TOK_FN, id, 0, line, startCol);
    }
    if (id == "return") {
        return Token(TOK_RETURN, id, 0, line, startCol);
    }
    
    // Otherwise it's an identifier
    return Token(TOK_IDENTIFIER, id, 0, line, startCol);
}

Token Lexer::nextToken() {
    skipWhitespace();
    
    if (pos >= input.size()) {
        return Token(TOK_EOF, "", 0, line, column);
    }
    
    char c = peek();
    
    if (std::isdigit(c)) {
        return number();
    }
    
    if (std::isalpha(c) || c == '_') {
        return identifier();
    }
    
    int startCol = column;
    advance();
    
    switch (c) {
        case '+': return Token(TOK_PLUS, "+", 0, line, startCol);
        case '-': return Token(TOK_MINUS, "-", 0, line, startCol);
        case '*': return Token(TOK_STAR, "*", 0, line, startCol);
        case '/': return Token(TOK_SLASH, "/", 0, line, startCol);
        case '=': return Token(TOK_ASSIGN, "=", 0, line, startCol);
        case ';': return Token(TOK_SEMICOLON, ";", 0, line, startCol);
        case '(': return Token(TOK_LPAREN, "(", 0, line, startCol);
        case ')': return Token(TOK_RPAREN, ")", 0, line, startCol);
        case '{': return Token(TOK_LBRACE, "{", 0, line, startCol);
        case '}': return Token(TOK_RBRACE, "}", 0, line, startCol);
        case ',': return Token(TOK_COMMA, ",", 0, line, startCol);
        default: return Token(TOK_UNKNOWN, std::string(1, c), 0, line, startCol);
    }
}

std::vector<Token> Lexer::tokenize() {
    std::vector<Token> tokens;
    Token tok;
    do {
        tok = nextToken();
        tokens.push_back(tok);
    } while (tok.type != TOK_EOF);
    return tokens;
}
