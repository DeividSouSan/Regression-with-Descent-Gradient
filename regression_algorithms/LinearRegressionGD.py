import random


def LinearRegressionGD(
        x_values: list,
        y_values: list,
        batch: int,
        learning_rate: float,
        max_steps: int,
        initial_params: list):

    sample_size = len(x_values)

    # Hyphotesis
    def h(x, b0=1, b1=1):
        """
        The hypothesis of a linear regression is always of the form:
            h(x) = b0 + (b1 * x)

        This functios return the h(x) for the params b0 and b1 provided
        """
        return b0 + (b1 * x)

    # Cost Function Mean Squared Error
    def J(b0, b1):
        """
        The cost function measures the error of the hypothesis result in relation to the expected values.

        The cost function used is Mean Squared Error. 

        The size of the batch is defined in the params of the LinearRegressionGD.

        return int
        """
        squared_errors = 0

        # Selecting just a part of the samples
        samples_index = random.sample(list(range(sample_size)), batch)
        x_sample, y_sample = [], []
        for index in samples_index:
            x_sample.append(x_values[index])
            y_sample.append(y_values[index])

        for index in range(batch):
            squared_errors += (h(x_sample[index],
                               b0, b1) - y_sample[index]) ** 2

        return squared_errors / batch

    # Derivadas Parciais da Função de Custo

    def dJb0(b0, b1):
        """
        Parial derivative of J(b0, b1) in relation to b0
        """
        samples_index = random.sample(list(range(sample_size)), batch)
        x_sample, y_sample = [], []
        for index in samples_index:
            x_sample.append(x_values[index])
            y_sample.append(y_values[index])

        rows_sum = 0
        for index in range(batch):
            rows_sum += h(x_sample[index], b0, b1) - y_sample[index]

        return rows_sum / batch

    # Parial derivative of J(b0, b1) in relation to b1

    def dJb1(b0, b1):
        """
        Parial derivative of J(b0, b1) in relation to b0
        """
        samples_index = random.sample(list(range(sample_size)), batch)
        x_sample, y_sample = [], []
        for index in samples_index:
            x_sample.append(x_values[index])
            y_sample.append(y_values[index])

        rows_sum = 0
        for index in range(batch):
            rows_sum += (h(x_sample[index], b0, b1) -
                         y_sample[index]) * x_sample[index]

        return rows_sum / batch

    b0, b1 = 0, 0
    b0_prev, b1_prev = initial_params
    for step in range(max_steps):
        b0_new = b0_prev - (learning_rate * dJb0(b0_prev, b1_prev))
        b1_new = b1_prev - (learning_rate * dJb1(b0_prev, b1_prev))

        b0_minimized = round(b0_new, 10) == round(b0_prev, 10)
        b1_minimized = round(b1_new, 10) == round(b1_prev, 10)
        if b0_minimized and b1_minimized:
            b0 = b0_new
            b1 = b1_new
            break

        elif step + 1 == max_steps:
            b0 = b0_new
            b1 = b1_new

        b0_prev = b0_new
        b1_prev = b1_new

    retornar um dicionario com essas informações
    return b0, b1, step, J(b0, b1)
