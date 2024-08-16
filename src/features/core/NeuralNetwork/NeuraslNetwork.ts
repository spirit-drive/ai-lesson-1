import mnist from 'mnist';

export class NeuralNetwork {
  weightsInputHidden;
  weightsHiddenOutput;
  biasHidden: number[];
  biasOutput: number[];
  learningRate: number;
  hiddenInput: number[];
  hiddenOutput: number[];
  finalInput: number[];
  finalOutput: number[];
  constructor(inputSize: number, hiddenSize: number, outputSize: number) {
    // Инициализация весов
    this.weightsInputHidden = this.initWeights(inputSize, hiddenSize);
    this.weightsHiddenOutput = this.initWeights(hiddenSize, outputSize);
    this.biasHidden = Array(hiddenSize).fill(0);
    this.biasOutput = Array(outputSize).fill(0);
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
    // Скрытый слой
    this.hiddenInput = this.addBias(this.dotProduct(input, this.weightsInputHidden), this.biasHidden);
    this.hiddenOutput = this.relu(this.hiddenInput);

    // Выходной слой
    this.finalInput = this.addBias(this.dotProduct(this.hiddenOutput, this.weightsHiddenOutput), this.biasOutput);
    this.finalOutput = this.softmax(this.finalInput);

    return this.finalOutput;
  }

  // Обратное распространение
  backward(input: number[], target: number[]) {
    // Выходной слой
    const outputErrors = this.finalOutput.map((o, i) => o - target[i]);
    const hiddenErrors = this.dotProductTransposed(outputErrors, this.weightsHiddenOutput);

    // Обновляем веса и смещения между скрытым и выходным слоем
    this.updateWeights(this.weightsHiddenOutput, this.hiddenOutput, outputErrors, this.biasOutput);

    // Производная ReLU для скрытого слоя
    const hiddenGrad = this.reluDerivative(this.hiddenInput).map((g, i) => g * hiddenErrors[i]);

    // Обновляем веса и смещения между входным и скрытым слоем
    this.updateWeights(this.weightsInputHidden, input, hiddenGrad, this.biasHidden);
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
        sum += vector[i] * matrixB[i][j];
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
export const nn = new NeuralNetwork(784, 784, 10); // 784 входов (28x28 пикселей), 128 скрытых нейронов, 10 выходов (цифры 0-9)

const { training, test } = mnist.set(8000, 2000);

export { training, test };

for (let i = 0; i < training.length; i++) {
  const input = training[i].input;
  const target = training[i].output;
  nn.train(input, target);
}

let right = 0;
let mistake = 0;

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
console.log({ mistake, right }, right / test.length);
