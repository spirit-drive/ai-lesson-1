import mnist from 'mnist';

export class NeuralNetwork {
  weights: number[][][];
  biases: number[][];
  learningRate: number;
  inputs: number[][];
  outputs: number[][];
  constructor(...layers: number[]) {
    // Инициализация весов
    this.weights = [];
    this.biases = [];
    this.inputs = [];
    this.outputs = [];
    for (let i = 0; i < layers.length - 1; i++) {
      this.weights.push(this.initWeights(layers[i], layers[i + 1]));
      this.biases.push(Array(layers[i + 1]).fill(0));
    }
    this.learningRate = 0.01;
  }

  // Инициализация весов случайными значениями
  initWeights(inputSize: number, outputSize: number) {
    return Array.from({ length: inputSize }, () => Array.from({ length: outputSize }, () => Math.random() * 2 - 1));
  }

  // Активационная функция ReLU
  relu(x: number[]) {
    return x.map((v) => Math.max(0, v));
  }

  // Производная ReLU
  reluDerivative(x: number[]) {
    return x.map((v) => (v > 0 ? 1 : 0));
  }

  // Softmax для вывода вероятностей
  softmax(output: number[]) {
    const expValues = output.map(Math.exp);
    const sumExpValues = expValues.reduce((acc, val) => acc + val, 0);
    return expValues.map((val) => val / sumExpValues);
  }

  // Прямое распространение
  forward(input: number[]) {
    let vector = input;
    for (let i = 0; i < this.biases.length - 1; i++) {
      // Скрытые слои
      vector = this.inputs[i] = this.addBias(this.dotProduct(vector, this.weights[i]), this.biases[i]);
      this.outputs[i] = this.relu(vector);
    }

    const lastIndex = this.biases.length - 1;

    // Выходной слой
    this.inputs[lastIndex] = this.addBias(
      this.dotProduct(this.outputs[lastIndex - 1], this.weights[lastIndex]),
      this.biases[lastIndex]
    );
    this.outputs[lastIndex] = this.softmax(this.inputs[lastIndex]);
    return this.outputs[lastIndex];
  }

  // Обратное распространение
  backward(input: number[], target: number[]) {
    let outputErrors = this.outputs[this.outputs.length - 1].map((o, i) => o - target[i]);

    for (let i = this.weights.length - 1; i > 0; i--) {
      // Выходной слой
      const layerErrors = this.dotProductTransposed(outputErrors, this.weights[i]);

      // Обновляем веса и смещения между скрытым и выходным слоем
      this.updateWeights(this.weights[i], this.outputs[i - 1], outputErrors, this.biases[i]);
      outputErrors = layerErrors;
    }

    // Производная ReLU для скрытого слоя
    const hiddenGrad = this.reluDerivative(this.inputs[0]).map((g, i) => g * outputErrors[i]);
    // Обновляем веса и смещения между входным и скрытым слоем
    this.updateWeights(this.weights[0], input, hiddenGrad, this.biases[0]);
  }

  // Функция для обновления весов и смещений
  updateWeights(weights: number[][], layerOutput: number[], errors: number[], bias: number[]) {
    for (let i = 0; i < weights.length; i++) {
      for (let j = 0; j < weights[0].length; j++) {
        weights[i][j] -= this.learningRate * errors[j] * layerOutput[i];
      }
    }
    // Обновление смещений
    for (let j = 0; j < bias.length; j++) {
      bias[j] -= this.learningRate * errors[j];
    }
  }

  // Умножение матриц
  dotProduct(vector: number[], matrixB: number[][]) {
    const result: number[] = [];

    const colsB = matrixB[0].length;

    for (let j = 0; j < colsB; j++) {
      let sum = 0;
      for (let i = 0; i < vector.length; i++) {
        sum += vector[i] * matrixB[i]?.[j] || 0;
      }
      result[j] = sum;
    }

    return result;
  }

  // Транспонированное умножение матриц
  dotProductTransposed(errors: number[], weights: number[][]) {
    return weights.map((_, i) => errors.reduce((acc, e, j) => acc + e * weights[i][j], 0));
  }

  // Добавляем смещение
  addBias(inputs: number[], bias: number[]) {
    return inputs.map((value, i) => value + bias[i]);
  }

  // Функция обучения сети
  train(input: number[], target: number[]) {
    this.forward(input);
    this.backward(input, target);
  }
}

// Пример использования сети
export const nn = new NeuralNetwork(784, 128, 10); // 784 входов (28x28 пикселей), 128 скрытых нейронов, 10 выходов (цифры 0-9)

const { training, test } = mnist.set(8000, 2000);

export { training, test };

console.time('training');
for (let i = 0; i < training.length; i++) {
  const input = training[i].input;
  const target = training[i].output;
  nn.train(input, target);
}
console.timeEnd('training');

let right = 0;
let mistake = 0;

console.time('test');
for (let i = 0; i < test.length; i++) {
  const input = test[i].input;
  const target = test[i].output;
  const prediction = nn.forward(input); // testInput — это новое изображение для распознавания
  let index = 0;
  let max = prediction[0];
  for (let j = 1; j < prediction.length; j++) {
    if (max < prediction[j]) {
      index = j;
      max = prediction[j];
    }
  }
  if (target[index]) {
    right++;
  } else {
    mistake++;
  }
}
console.timeEnd('test');
console.log({ mistake, right }, right / test.length);
