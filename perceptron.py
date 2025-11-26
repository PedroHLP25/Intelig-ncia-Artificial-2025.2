# Função de treinamento do perceptron
def perceptron_train(examples, labels, weights, alpha, n_epochs,
                     log_path='perceptron_steps.txt'):
    with open(log_path, 'w', encoding='utf-8') as f:
        for epoch in range(n_epochs):
            f.write(f'Ciclo {epoch + 1}\n')
            for i, (x, y) in enumerate(zip(examples, labels)):
                s = sum([w * xj for w, xj in zip(weights, x)])
                pred = 1 if s > 0 else 0
                error = y - pred

                # linha principal
                f.write(
                    f'Example {i + 1}: x={x} y={y} | '
                    f'sum={s:.2f} pred={pred} error={error}\n'
                )

                # pesos antes
                f.write(f'  Pesos antes: {weights}\n')

                # atualização
                for j in range(len(weights)):
                    weights[j] += alpha * error * x[j]

                # pesos depois
                f.write(f'  Pesos depois:{weights}\n\n')

        f.write(f'Pesos finais: {weights}\n')
    return weights


# Dados do exercício
# x0 = 1 (bias), x1 = estudou, x2 = fez trabalho
examples = [
    [1, 0, 0],  # Joaozinho
    [1, 0, 1],  # Huguinho
    [1, 1, 0],  # Zezinho
    [1, 1, 1],  # Luizinho
]
labels = [0, 0, 1, 1]

weights = [0.0, 0.0, 0.0]
alpha = 0.1
n_epochs = 2

# Treinamento e geração do TXT
final_weights = perceptron_train(examples, labels, weights.copy(), alpha, n_epochs)
print(final_weights)
