#include "Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::mini;

MiniDialect::MiniDialect(MLIRContext *context) 
    : Dialect(getDialectNamespace(), context, TypeID::get<MiniDialect>()) {
    initialize();
}

void MiniDialect::initialize() {
    addOperations<ConstantOp, AddOp, PrintOp>();
}

// ConstantOp Implementation
void ConstantOp::build(OpBuilder &builder, OperationState &state, double value) {
    auto dataType = builder.getIntegerType(32);
    auto dataAttribute = builder.getI32IntegerAttr(static_cast<int32_t>(value));
    state.addAttribute("value", dataAttribute);
    state.addTypes(dataType);
}

double ConstantOp::getValue() {
    return (*this)->getAttrOfType<IntegerAttr>("value").getInt();
}

/*ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
    IntegerAttr valueAttr;
    if (parser.parseAttribute(valueAttr, "value", result.attributes))
        return failure();
    result.addTypes(parser.getBuilder().getIntegerType(32));
    return success();
}*/

void ConstantOp::print(OpAsmPrinter &p) {
    p << " ";
    p << getValue();
    p << " : ";
    p << (*this)->getResult(0).getType();
}

/*LogicalResult ConstantOp::verify() {
    return success();
}*/

// AddOp Implementation
void AddOp::build(OpBuilder &builder, OperationState &state, Value lhs, Value rhs) {
    state.addOperands({lhs, rhs});
    state.addTypes(builder.getIntegerType(32));
}

/*ParseResult AddOp::parse(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::UnresolvedOperand lhs, rhs;
    Type type;
    
    if (parser.parseOperand(lhs) || parser.parseComma() || 
        parser.parseOperand(rhs) || parser.parseColonType(type))
        return failure();
    
    if (parser.resolveOperands({lhs, rhs}, type, result.operands))
        return failure();
    
    result.addTypes(type);
    return success();
}*/

void AddOp::print(OpAsmPrinter &p) {
    p << " ";
    p << (*this)->getOperand(0);
    p << ", ";
    p << (*this)->getOperand(1);
    p << " : ";
    p << (*this)->getResult(0).getType();
}

/*LogicalResult AddOp::verify() {
    if ((*this)->getNumOperands() != 2)
        return emitOpError("requires exactly 2 operands");
    return success();
}*/

// PrintOp Implementation
void PrintOp::build(OpBuilder &builder, OperationState &state, Value input) {
    state.addOperands(input);
}

/*ParseResult PrintOp::parse(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::UnresolvedOperand operand;
    Type type;
    
    if (parser.parseOperand(operand) || parser.parseColonType(type))
        return failure();
    
    if (parser.resolveOperand(operand, type, result.operands))
        return failure();
    
    return success();
}*/

void PrintOp::print(OpAsmPrinter &p) {
    p << " ";
    p << (*this)->getOperand(0);
    p << " : ";
    p << (*this)->getOperand(0).getType();
}

/*LogicalResult PrintOp::verify() {
    if ((*this)->getNumOperands() != 1)
        return emitOpError("requires exactly 1 operand");
    return success();
}*/
