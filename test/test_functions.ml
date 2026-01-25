fn add(x, y) {
    return x + y;
}

fn square(n) {
    return n * n;
}

fn sumOfSquares(a, b) {
    x = square(a);
    y = square(b);
    return add(x, y);
}

result = add(3, 5);
print result;

sq = square(7);
print sq;

sos = sumOfSquares(3, 4);
print sos;
