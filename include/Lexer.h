#ifndef LEXER_H
#define LEXER_H

#include "Token.h"
#include <string>
#include <vector>

class Lexer {
private:
    std::string input;
    size_t pos;
    int line;
    int column;
    
    char peek();
    char advance();
    void skipWhitespace();
    Token number();
    Token identifier();

public:
    Lexer(const std::string& source);
    Token nextToken();
    std::vector<Token> tokenize();
};

#endif // LEXER_H
