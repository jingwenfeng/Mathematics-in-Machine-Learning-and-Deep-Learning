# NN by hand
import math
import random


class ActivationFunction:
    """
    Activation function f and derivative f'.
    Supported: "sigmoid", "tanh", "relu", "linear"
    """

    def __init__(self, name):
        self.name = name.lower()
        if self.name not in ("sigmoid", "tanh", "relu", "linear"):
            raise ValueError("Unsupported activation function: " + str(name))

    def value(self, x):
        if self.name == "sigmoid":
            return 1.0 / (1.0 + math.exp(-x))
        elif self.name == "tanh":
            return math.tanh(x)
        elif self.name == "relu":
            return x if x > 0.0 else 0.0
        elif self.name == "linear":
            return x

    def derivative(self, x):
        if self.name == "sigmoid":
            s = self.value(x)
            return s * (1.0 - s)
        elif self.name == "tanh":
            t = math.tanh(x)
            return 1.0 - t * t
        elif self.name == "relu":
            return 1.0 if x > 0.0 else 0.0
        elif self.name == "linear":
            return 1.0


class NeuralNetwork:
    """
    Fully-connected feed-forward neural network.

    layer_sizes: [n_input, n_hidden1, ..., n_output]
    activation_names: one string or list of strings (per layer with weights).
    """

    def __init__(self, layer_sizes, activation_names, random_seed=None):
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must be at least [n_in, n_out].")
        self.layer_sizes = list(layer_sizes)
        self.num_layers = len(layer_sizes) - 1  # number of weight layers

        # Handle activation names
        if isinstance(activation_names, str):
            activation_names = [activation_names] * self.num_layers
        else:
            activation_names = list(activation_names)

        if len(activation_names) == 1 and self.num_layers > 1:
            activation_names = activation_names * self.num_layers

        if len(activation_names) != self.num_layers:
            raise ValueError(
                "activation_names must have length 1 or (len(layer_sizes) - 1). "
                "Got %d for %d layers."
                % (len(activation_names), self.num_layers)
            )

        self.activation_functions = [
            ActivationFunction(name) for name in activation_names
        ]

        if random_seed is not None:
            random.seed(random_seed)

        # Initialize weights and biases
        # weights[l] shape: (layer_sizes[l+1], layer_sizes[l])
        # biases[l] shape: (layer_sizes[l+1],)
        self.weights = []
        self.biases = []

        for l in range(self.num_layers):
            n_in = layer_sizes[l]
            n_out = layer_sizes[l + 1]
            w = [
                [random.uniform(-1.0, 1.0) for _ in range(n_in)]
                for __ in range(n_out)
            ]
            b = [random.uniform(-1.0, 1.0) for __ in range(n_out)]
            self.weights.append(w)
            self.biases.append(b)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, input_vector):
        """
        Forward pass: compute all activations a^(l) and pre-activations z^(l).

        Returns:
            activations: [a^(0), ..., a^(L)]
            zs:          [None, z^(1), ..., z^(L)]
        """
        if len(input_vector) != self.layer_sizes[0]:
            raise ValueError(
                "Input dimension mismatch. Expected %d, got %d."
                % (self.layer_sizes[0], len(input_vector))
            )

        activations = [list(input_vector)]  # a^(0)
        zs = [None]                         # no z for input layer

        a = list(input_vector)
        for l in range(self.num_layers):
            w = self.weights[l]
            b = self.biases[l]
            fn = self.activation_functions[l]

            z_layer = []
            a_layer = []
            for i in range(len(w)):
                # z^(l+1)_i = sum_j w^(l+1)_{i,j} * a^(l)_j + b^(l+1)_i
                z_i = 0.0
                for j in range(len(a)):
                    z_i += w[i][j] * a[j]
                z_i += b[i]
                # a^(l+1)_i = f(z^(l+1)_i)
                a_i = fn.value(z_i)
                z_layer.append(z_i)
                a_layer.append(a_i)

            zs.append(z_layer)
            activations.append(a_layer)
            a = a_layer

        return activations, zs

    # ------------------------------------------------------------------
    # Loss: Mean Squared Error
    # ------------------------------------------------------------------
    @staticmethod
    def mean_squared_error(output_activations, target_vector):
        """
        L = 1/2 * sum_k (a_k - y_k)^2
        """
        if len(output_activations) != len(target_vector):
            raise ValueError("Output/target dimension mismatch.")
        loss = 0.0
        for a_k, y_k in zip(output_activations, target_vector):
            diff = a_k - y_k
            loss += 0.5 * diff * diff
        return loss

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------
    def backward(self, activations, zs, target_vector):
        """
        Backpropagation for one sample.

        Returns:
            nabla_w: gradients of weights, same shape as self.weights
            nabla_b: gradients of biases, same shape as self.biases
            deltas:  δ^(l) for each layer (index 0 is None)
        """
        L = self.num_layers
        deltas = [None] * (L + 1)

        # Output layer delta
        output_activations = activations[L]
        if len(output_activations) != len(target_vector):
            raise ValueError("Target dimension mismatch.")

        delta_L = []
        fn_L = self.activation_functions[L - 1]
        for k in range(len(output_activations)):
            a_L_k = output_activations[k]
            y_k = target_vector[k]
            z_L_k = zs[L][k]

            dL_da = a_L_k - y_k           # ∂L/∂a^(L)_k
            fprime = fn_L.derivative(z_L_k)
            delta_k = dL_da * fprime      # δ^(L)_k
            delta_L.append(delta_k)

        deltas[L] = delta_L

        # Hidden layers deltas
        for l in range(L - 1, 0, -1):
            fn_l = self.activation_functions[l - 1]
            delta_next = deltas[l + 1]
            w_next = self.weights[l]
            n_l = self.layer_sizes[l]
            n_next = self.layer_sizes[l + 1]

            delta_l = []
            for j in range(n_l):
                # δ^(l)_j = f'(z^(l)_j) * sum_k w^(l+1)_{k,j} * δ^(l+1)_k
                s = 0.0
                for k in range(n_next):
                    s += w_next[k][j] * delta_next[k]
                z_l_j = zs[l][j]
                fprime = fn_l.derivative(z_l_j)
                delta_j = fprime * s
                delta_l.append(delta_j)
            deltas[l] = delta_l

        # Gradients
        nabla_w = []
        nabla_b = []
        for l in range(1, L + 1):
            delta_l = deltas[l]        # δ^(l)
            a_prev = activations[l - 1]  # a^(l-1)

            grad_w_layer = []
            grad_b_layer = []
            for i in range(len(delta_l)):
                delta_i = delta_l[i]
                # ∂L/∂b^(l)_i = δ^(l)_i
                grad_b_layer.append(delta_i)
                # ∂L/∂w^(l)_{i,j} = δ^(l)_i * a^(l-1)_j
                grad_w_row = []
                for j in range(len(a_prev)):
                    grad_w_row.append(delta_i * a_prev[j])
                grad_w_layer.append(grad_w_row)

            nabla_w.append(grad_w_layer)
            nabla_b.append(grad_b_layer)

        return nabla_w, nabla_b, deltas

    # ------------------------------------------------------------------
    # Gradient descent update
    # ------------------------------------------------------------------
    def apply_gradients(self, nabla_w, nabla_b, learning_rate):
        """
        w := w - eta * ∂L/∂w
        b := b - eta * ∂L/∂b
        """
        for l in range(self.num_layers):
            for i in range(len(self.weights[l])):
                for j in range(len(self.weights[l][i])):
                    self.weights[l][i][j] -= learning_rate * nabla_w[l][i][j]
                self.biases[l][i] -= learning_rate * nabla_b[l][i]

    # ------------------------------------------------------------------
    # SGD training
    # ------------------------------------------------------------------
    def train_sgd(
        self,
        training_data,
        epochs,
        learning_rate,
        print_every=1,
        print_math=False,
    ):
        """
        Train using Stochastic Gradient Descent (SGD).

        training_data: list of (x, y)
        epochs:        number of passes over training_data
        learning_rate: η (step size)
        print_every:   print math details every N steps
        print_math:    if True, show formulas + numeric steps
        """
        step = 0
        for epoch in range(epochs):
            random.shuffle(training_data)
            for x, y in training_data:
                step += 1

                activations, zs = self.forward(x)
                loss = self.mean_squared_error(activations[-1], y)
                nabla_w, nabla_b, deltas = self.backward(activations, zs, y)

                if print_math and (step % print_every == 0):
                    self._print_step_debug(
                        step=step,
                        epoch=epoch,
                        x=x,
                        y=y,
                        activations=activations,
                        zs=zs,
                        deltas=deltas,
                        nabla_w=nabla_w,
                        nabla_b=nabla_b,
                        learning_rate=learning_rate,
                        loss=loss,
                    )

                self.apply_gradients(nabla_w, nabla_b, learning_rate)

    # ------------------------------------------------------------------
    # Detailed math printing (one SGD step)
    # ------------------------------------------------------------------
    def _print_step_debug(
        self,
        step,
        epoch,
        x,
        y,
        activations,
        zs,
        deltas,
        nabla_w,
        nabla_b,
        learning_rate,
        loss,
    ):
        """Print formulas and numeric steps for one SGD update."""

        def fmt(v):
            return "{:.6f}".format(v)

        print("============================================================")
        print("SGD STEP %d (epoch %d)" % (step, epoch))
        print("------------------------------------------------------------")
        print("Input a^(0) =", [fmt(v) for v in x])
        print("Target y    =", [fmt(v) for v in y])
        print()

        # -------- FORWARD PASS --------
        print("FORWARD PASS")
        print("------------")
        for l in range(1, self.num_layers + 1):
            fn = self.activation_functions[l - 1]
            print("Layer %d (activation: %s)" % (l, fn.name))
            a_prev = activations[l - 1]
            z_l = zs[l]
            a_l = activations[l]
            for i in range(len(a_l)):
                print("  Neuron i = %d" % i)
                print("    z^(%d)_%d = sum_j w^(%d)_%d,j * a^(%d)_j + b^(%d)_%d"
                      % (l, i, l, i, l - 1, l, i))
                for j in range(len(a_prev)):
                    w_ij = self.weights[l - 1][i][j]
                    print("      w^(%d)_%d,%d * a^(%d)_%d = %s * %s"
                          % (l, i, j, l - 1, j, fmt(w_ij), fmt(a_prev[j])))
                b_i = self.biases[l - 1][i]
                print("      + b^(%d)_%d = %s" % (l, i, fmt(b_i)))
                print("    => z^(%d)_%d = %s" % (l, i, fmt(z_l[i])))
                print("    a^(%d)_%d = f_%d(z^(%d)_%d) = %s(%s) = %s"
                      % (l, i, l, l, i, fn.name, fmt(z_l[i]), fmt(a_l[i])))
                print()

        out = activations[-1]
        print("Output a^(L) =", [fmt(v) for v in out])
        print()

        # -------- LOSS --------
        print("LOSS (Mean Squared Error)")
        print("-------------------------")
        print("L = 1/2 * sum_k (a^(L)_k - y_k)^2")
        for k in range(len(out)):
            diff = out[k] - y[k]
            term = 0.5 * diff * diff
            print("  k=%d: 1/2 * (%s - %s)^2 = %s"
                  % (k, fmt(out[k]), fmt(y[k]), fmt(term)))
        print("=> L =", fmt(loss))
        print()

        # -------- BACKWARD PASS --------
        print("BACKWARD PASS")
        print("-------------")
        L = self.num_layers
        print("Output layer (l = %d)" % L)
        fn_L = self.activation_functions[L - 1]
        for k in range(len(out)):
            a_L_k = out[k]
            y_k = y[k]
            z_L_k = zs[L][k]
            dL_da = a_L_k - y_k
            fprime = fn_L.derivative(z_L_k)
            delta_k = deltas[L][k]
            print("  k = %d" % k)
            print("    dL/da^(L)_%d = a^(L)_%d - y_%d = %s - %s = %s"
                  % (k, k, k, fmt(a_L_k), fmt(y_k), fmt(dL_da)))
            print("    f'(z^(L)_%d) = f'(%s) = %s"
                  % (k, fmt(z_L_k), fmt(fprime)))
            print("    delta^(L)_%d = dL/da^(L)_%d * f'(z^(L)_%d) = %s * %s = %s"
                  % (k, k, k, fmt(dL_da), fmt(fprime), fmt(delta_k)))
        print()

        print("Hidden layers")
        for l in range(L - 1, 0, -1):
            fn_l = self.activation_functions[l - 1]
            print("Layer l = %d" % l)
            delta_next = deltas[l + 1]
            w_next = self.weights[l]
            n_l = self.layer_sizes[l]
            n_next = self.layer_sizes[l + 1]
            for j in range(n_l):
                print("  Neuron j = %d" % j)
                print("    delta^(%d)_%d = f'(z^(%d)_%d) * sum_k w^(%d)_k,%d * delta^(%d)_k"
                      % (l, j, l, j, l + 1, j, l + 1))
                sum_val = 0.0
                for k in range(n_next):
                    w_kj = w_next[k][j]
                    d_next_k = delta_next[k]
                    print("      w^(%d)_%d,%d * delta^(%d)_%d = %s * %s"
                          % (l + 1, k, j, l + 1, k, fmt(w_kj), fmt(d_next_k)))
                    sum_val += w_kj * d_next_k
                z_l_j = zs[l][j]
                fprime = fn_l.derivative(z_l_j)
                delta_j = deltas[l][j]
                print("      sum_k = %s" % fmt(sum_val))
                print("      f'(z^(%d)_%d) = f'(%s) = %s"
                      % (l, j, fmt(z_l_j), fmt(fprime)))
                print("    => delta^(%d)_%d = %s * %s = %s"
                      % (l, j, fmt(fprime), fmt(sum_val), fmt(delta_j)))
            print()

        # -------- GRADIENTS --------
        print("GRADIENTS")
        print("---------")
        for l in range(1, L + 1):
            print("Layer %d:" % l)
            delta_l = deltas[l]
            a_prev = activations[l - 1]
            for i in range(len(delta_l)):
                delta_i = delta_l[i]
                print("  Neuron i = %d" % i)
                print("    dL/db^(%d)_%d = delta^(%d)_%d = %s"
                      % (l, i, l, i, fmt(nabla_b[l - 1][i])))
                print("    dL/dw^(%d)_%d,j = delta^(%d)_%d * a^(%d)_j"
                      % (l, i, l, i, l - 1))
                for j in range(len(a_prev)):
                    grad_ij = nabla_w[l - 1][i][j]
                    print("      j = %d: %s * %s = %s"
                          % (j, fmt(delta_i), fmt(a_prev[j]), fmt(grad_ij)))
            print()

        # -------- UPDATE RULE --------
        print("UPDATE STEP")
        print("-----------")
        print("Learning rate eta =", fmt(learning_rate))
        print("w := w - eta * dL/dw")
        print("b := b - eta * dL/db")
        print("============================================================")
        print()
