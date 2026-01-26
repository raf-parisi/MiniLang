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

## Esempio

**Codice sorgente (test_if.ml):**
```perl
fn compare(a, b) {
    if (a > b) {
        print a;
    } else {
        print b;
    }
    return 0;
}

x = 10;
y = 5;

if (x > y) {
    result = x + y;
    print result;
} else {
    result = x - y;
    print result;
}
```

**MLIR generato (Mini Dialect):**
```mlir
mini.func @compare(i32, i32) -> i32 {
  %0 = mini.cmp "sgt", %arg0, %arg1 : i1
  scf.if %0 {
    mini.print %arg0 : i32
  } else {
    mini.print %arg1 : i32
  }
  %1 = mini.constant 0 : i32
  mini.return %1 : i32
}

mini.func @main() -> i32 {
  %0 = mini.constant 10 : i32
  %1 = mini.constant 5 : i32
  %2 = mini.cmp "sgt", %0, %1 : i1
  scf.if %2 {
    %3 = mini.add %0, %1 : i32
    mini.print %3 : i32
  } else {
    %4 = mini.sub %0, %1 : i32
    mini.print %4 : i32
  }
  %5 = mini.constant 0 : i32
  mini.return %5 : i32
}
```

**LLVM IR finale:**
```llvm
define i32 @compare(i32 %0, i32 %1) {
  %3 = icmp sgt i32 %0, %1
  br i1 %3, label %4, label %6
4:
  %5 = call i32 (ptr, ...) @printf(ptr @fmt, i32 %0)
  br label %8
6:
  %7 = call i32 (ptr, ...) @printf(ptr @fmt, i32 %1)
  br label %8
8:
  ret i32 0
}

define i32 @main() {
  br i1 true, label %1, label %3
1:
  %2 = call i32 (ptr, ...) @printf(ptr @fmt, i32 15)
  br label %5
3:
  %4 = call i32 (ptr, ...) @printf(ptr @fmt, i32 5)
  br label %5
5:
  ret i32 0
}
```

## Features

- ✅ Aritmetica intera (add, sub, mul, div)
- ✅ Variabili e assegnamenti
- ✅ Funzioni con parametri
- ✅ Operatori di confronto (<, >, ==, !=)
- ✅ Control flow (if-else)
- ✅ Print statement

## Requisiti

- LLVM 18+
- MLIR (incluso in LLVM)
- CMake 3.20+
- Clang/GCC con supporto C++17

```
