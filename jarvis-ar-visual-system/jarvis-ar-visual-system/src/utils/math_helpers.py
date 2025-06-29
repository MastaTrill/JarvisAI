def vector_addition(v1: list[float], v2: list[float]) -> list[float]:
    """Adds two vectors element-wise.

    Args:
        v1 (list[float]): The first vector.
        v2 (list[float]): The second vector.

    Returns:
        list[float]: The resulting vector after addition.
    """
    return [a + b for a, b in zip(v1, v2)]


def vector_subtraction(v1: list[float], v2: list[float]) -> list[float]:
    """Subtracts the second vector from the first vector element-wise.

    Args:
        v1 (list[float]): The first vector.
        v2 (list[float]): The second vector.

    Returns:
        list[float]: The resulting vector after subtraction.
    """
    return [a - b for a, b in zip(v1, v2)]


def dot_product(v1: list[float], v2: list[float]) -> float:
    """Calculates the dot product of two vectors.

    Args:
        v1 (list[float]): The first vector.
        v2 (list[float]): The second vector.

    Returns:
        float: The dot product of the two vectors.
    """
    return sum(a * b for a, b in zip(v1, v2))


def cross_product(v1: list[float], v2: list[float]) -> list[float]:
    """Calculates the cross product of two 3D vectors.

    Args:
        v1 (list[float]): The first 3D vector.
        v2 (list[float]): The second 3D vector.

    Returns:
        list[float]: The resulting vector from the cross product.
    """
    return [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ]


def matrix_multiplication(m1: list[list[float]], m2: list[list[float]]) -> list[list[float]]:
    """Multiplies two matrices.

    Args:
        m1 (list[list[float]]): The first matrix.
        m2 (list[list[float]]): The second matrix.

    Returns:
        list[list[float]]: The resulting matrix after multiplication.
    """
    return [
        [sum(a * b for a, b in zip(row, col)) for col in zip(*m2)]
        for row in m1
    ]


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    """Transposes a matrix.

    Args:
        matrix (list[list[float]]): The matrix to transpose.

    Returns:
        list[list[float]]: The transposed matrix.
    """
    return [list(row) for row in zip(*matrix)]