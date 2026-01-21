# MiniLang

Compilatore minimale da linguaggio custom a LLVM IR, costruito con MLIR.

> **🚧 Work In Progress** - Progetto educativo in sviluppo attivo

## Stato Attuale

### ✅ Implementato
- Lexer e Parser per espressioni aritmetiche
- Dialetto MLIR custom (`mini`)
- Lowering a LLVM IR
- Operazione: `print(addizioni)`

### 🔜 Da Implementare
- Control flow (if/else, while)
- Definizione e chiamata funzioni
- Variabili e assignment

## Esempio

**Input:**
```javascript
5 + 3;
```

**Output MLIR:**
```mlir
%0 = mini.constant 5 : i32
%1 = mini.constant 3 : i32
%2 = mini.add %0, %1 : i32
mini.print %2 : i32
```

## Build
```bash
git clone https://github.com/raf-parisi/MiniLang.git
cd MiniLang
mkdir build && cd build
cmake ..
make
./minilang ../test/simple.ml
```

## Architettura
```
Source → Lexer → Parser → AST → MLIRGen → LLVMIRGen → LLVM IR
```

## Struttura Progetto
```
MiniLang/
├── include/     # Header files
├── src/         # Implementazioni
├── test/        # File di test
└── CMakeLists.txt
```

## Pipeline

1. **Lexer**: Tokenizzazione
2. **Parser**: Costruzione AST
3. **MLIRGen**: AST → dialetto mini
4. **WrapInMainPass**: Wrapping in func.func @main
5. **LLVMIRGen**: mini dialect → LLVM dialect
6. **Output**: LLVM IR

## Tecnologie

- **LLVM/MLIR** 17-18
- **CMake** >= 3.20
- **C++17**
