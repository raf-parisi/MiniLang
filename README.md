# MiniLang

Compilatore giocattolo basato sull'infrastruttura MLIR/LLVM. Implementazione dell'intera pipeline di compilazione: frontend (lexing, parsing), generazione di rappresentazioni intermedie, ottimizzazioni e lowering fino a LLVM IR eseguibile.

## Architettura
```
Source Code (.ml)
      ↓
   Lexer (Token stream)
      ↓
   Parser (AST)
      ↓
  MLIR Generator (Mini Dialect)
      ↓
  Mini → Func Dialect
      ↓
  SCF → Control Flow
      ↓
  Func/CF → LLVM Dialect
      ↓
  LLVM IR
      ↓
  Executable (via clang)
```

## Pipeline di Compilazione

1. **Frontend**: Lexer e Parser generano l'AST dal codice sorgente
2. **MLIR Generation**: AST → Mini Dialect (custom operations)
3. **Dialect Lowering**: Mini → Standard Func Dialect
4. **Control Flow Lowering**: SCF (Structured Control Flow) → CF (Control Flow)
5. **LLVM Lowering**: Func/CF → LLVM Dialect
6. **Code Generation**: LLVM IR → Native Code


## Features

- ✅ Aritmetica intera (add, sub, mul, div)
- ✅ Variabili e assegnamenti
- ✅ Funzioni con parametri
- ✅ Operatori di confronto (<, >, ==, !=)
- ✅ Control flow (if-else, while)
- ✅ Print statement

## Requisiti

- LLVM 18
- MLIR (incluso in LLVM)
- CMake 3.20+
- Clang/GCC con supporto C++17
