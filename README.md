# MiniLang

Un compilatore minimale che dimostra la pipeline completa di compilazione con MLIR e LLVM, dal codice sorgente all'eseguibile.

## Caratteristiche

- Espressioni aritmetiche semplici con addizione
- Dialetto MLIR custom (`mini`)
- Lowering completo a LLVM IR
- Generazione codice nativo tramite clang


MiniLang supporta semplici statement di stampa con espressioni aritmetiche:

```
print 5 + 3 + 2;
print 10 + 20 + 15;
print 100;
```

## Requisiti

- LLVM 18+ con MLIR
- CMake 3.20+
- Compilatore C++17
- clang (per compilazione finale)

## Esempio

File di input `test/valid.ml`:
```
print 5 + 3 + 2;
print 10 + 20 + 15;
print 100;
```

Output:
```
10
45
100
```

## Architettura

Il compilatore implementa una pipeline di compilazione completa:

1. **Lexer** (`src/Lexer.cpp`) - Tokenizzazione del codice sorgente
2. **Parser** (`src/Parser.cpp`) - Costruzione dell'Abstract Syntax Tree (AST)
3. **MLIRGen** (`src/MLIRGen.cpp`) - Conversione da AST a dialetto MLIR custom
4. **Wrapping** (`src/LLVMIRGen.cpp`) - Wrapping del codice nella funzione main
5. **Lowering** (`src/LLVMIRGen.cpp`) - Lowering da dialetto custom a dialetto LLVM
6. **Export** (`src/LLVMIRGen.cpp`) - Export a LLVM IR

### Dialetto MLIR Custom

Il dialetto `mini` include tre operazioni:

- `mini.constant` - Costanti intere
- `mini.add` - Operazione di addizione
- `mini.print` - Stampa su stdout

Esempio di output MLIR:
```mlir
module {
  func.func @main() {
    %0 = mini.constant 5 : i32
    %1 = mini.constant 3 : i32
    %2 = mini.add %0, %1 : i32
    mini.print %2 : i32
    return
  }
}
```

Dopo il lowering al dialetto LLVM e l'export, questo diventa LLVM IR valido che può essere compilato in codice nativo.


## Dettagli Implementativi

### Sistema dei Tipi
Tutti i valori sono interi a 32-bit (`i32`).

### Operazioni
- **Addizione**: Associativa a sinistra, processa gli operandi da sinistra a destra
- **Costanti**: Literal interi convertiti in i32
- **Print**: Stampa valori su stdout usando printf

### Lowering MLIR
Il dialetto custom `mini` viene abbassato al dialetto LLVM attraverso pattern rewriting:
- `mini.constant` → `llvm.mlir.constant`
- `mini.add` → `llvm.add`
- `mini.print` → `llvm.call @printf`

### Build
```bash
cd Minilang/
mkdir build
cd build
cmake ..
make


