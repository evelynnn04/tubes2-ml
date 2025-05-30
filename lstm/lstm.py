import numpy as np

def softmax_crossentropy_loss(predictions, targets):
    """
    Compute softmax cross-entropy loss and gradient

    Args:
        predictions: raw logits (batch_size, num_classes)
        targets: true class indices (batch_size,)

    Returns:
        loss: scalar loss value
        gradient: gradient w.r.t. predictions (batch_size, num_classes)
    """
    batch_size = predictions.shape[0]

    # Softmax probabilities
    exp_pred = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
    probabilities = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)

    # One hot encoding of targets
    one_hot = np.zeros_like(predictions)
    one_hot[np.arange(batch_size), targets] = 1

    epsilon = 1e-15
    probabilities = np.clip(probabilities, epsilon, 1 - epsilon)

    loss = -np.mean(np.sum(one_hot * np.log(probabilities), axis=1))

    # Gradient of softmax + cross-entropy
    gradient = (probabilities - one_hot) / batch_size

    return loss, gradient

class EmbeddingLayer:
    def __init__(self, embedding_matrix):
        self.embedding_matrix = embedding_matrix
        self.vocab_size, self.embedding_dim = embedding_matrix.shape

    def forward(self, X):
        self.input_shape = X.shape
        return self.embedding_matrix[X]

    def backward(self, dout):
        return None

class LSTMLayer:
    def __init__(self, input_dim, hidden_dim, return_sequences=False):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.return_sequences = return_sequences

        scale = 1.0 / np.sqrt(input_dim + hidden_dim)

        self.Wf = np.random.uniform(-scale, scale, (input_dim, hidden_dim))
        self.Wi = np.random.uniform(-scale, scale, (input_dim, hidden_dim))
        self.Wc = np.random.uniform(-scale, scale, (input_dim, hidden_dim))
        self.Wo = np.random.uniform(-scale, scale, (input_dim, hidden_dim))

        self.Uf = np.random.uniform(-scale, scale, (hidden_dim, hidden_dim))
        self.Ui = np.random.uniform(-scale, scale, (hidden_dim, hidden_dim))
        self.Uc = np.random.uniform(-scale, scale, (hidden_dim, hidden_dim))
        self.Uo = np.random.uniform(-scale, scale, (hidden_dim, hidden_dim))

        self.bf = np.ones(hidden_dim)
        self.bi = np.zeros(hidden_dim)
        self.bc = np.zeros(hidden_dim)
        self.bo = np.zeros(hidden_dim)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def tanh(self, x):
        return np.tanh(np.clip(x, -500, 500))

    def forward(self, X):
        batch_size, seq_len, _ = X.shape

        h = np.zeros((batch_size, self.hidden_dim))
        c = np.zeros((batch_size, self.hidden_dim))
        self.cache = {
            'X': X,
            'h_states': [h.copy()], 
            'c_states': [c.copy()], 
            'f_gates': [],
            'i_gates': [],
            'c_candidates': [],
            'o_gates': []
        }

        outputs = []

        for t in range(seq_len):
            x_t = X[:, t, :]  # (batch_size, input_dim)

            # Gates
            f_t = self.sigmoid(np.dot(x_t, self.Wf) + np.dot(h, self.Uf) + self.bf)
            i_t = self.sigmoid(np.dot(x_t, self.Wi) + np.dot(h, self.Ui) + self.bi)
            c_candidate = self.tanh(np.dot(x_t, self.Wc) + np.dot(h, self.Uc) + self.bc)
            o_t = self.sigmoid(np.dot(x_t, self.Wo) + np.dot(h, self.Uo) + self.bo)

            # Update cell state and hidden state
            c = f_t * c + i_t * c_candidate
            h = o_t * self.tanh(c)

            # Store for backward pass
            self.cache['h_states'].append(h.copy())
            self.cache['c_states'].append(c.copy())
            self.cache['f_gates'].append(f_t.copy())
            self.cache['i_gates'].append(i_t.copy())
            self.cache['c_candidates'].append(c_candidate.copy())
            self.cache['o_gates'].append(o_t.copy())

            if self.return_sequences:
                outputs.append(h.copy())

        if self.return_sequences:
            return np.stack(outputs, axis=1)  # (batch_size, seq_len, hidden_dim)
        else:
            return h  # (batch_size, hidden_dim) - last hidden state only

    def backward(self, dh_out, learning_rate=0.001):
        """
        dh_out: gradient from next layer
        - If return_sequences=False: (batch_size, hidden_dim)
        - If return_sequences=True: (batch_size, seq_len, hidden_dim)
        """
        X = self.cache['X']
        batch_size, seq_len, input_dim = X.shape

        # Initialize gradients
        dWf = np.zeros_like(self.Wf)
        dWi = np.zeros_like(self.Wi)
        dWc = np.zeros_like(self.Wc)
        dWo = np.zeros_like(self.Wo)

        dUf = np.zeros_like(self.Uf)
        dUi = np.zeros_like(self.Ui)
        dUc = np.zeros_like(self.Uc)
        dUo = np.zeros_like(self.Uo)

        dbf = np.zeros_like(self.bf)
        dbi = np.zeros_like(self.bi)
        dbc = np.zeros_like(self.bc)
        dbo = np.zeros_like(self.bo)

        # Initialize hidden and cell state gradients
        dh_next = np.zeros((batch_size, self.hidden_dim))
        dc_next = np.zeros((batch_size, self.hidden_dim))

        # Process gradients for each timestep in reverse order
        for t in reversed(range(seq_len)):
            x_t = X[:, t, :]
            h_prev = self.cache['h_states'][t] 
            c_prev = self.cache['c_states'][t]  

            f_t = self.cache['f_gates'][t]
            i_t = self.cache['i_gates'][t]
            c_candidate = self.cache['c_candidates'][t]
            o_t = self.cache['o_gates'][t]
            c_t = self.cache['c_states'][t + 1]  # Current c state

            # Gradient from output
            if self.return_sequences:
                dh_t = dh_out[:, t, :] + dh_next
            else:
                dh_t = dh_out if t == seq_len - 1 else dh_next

            # Gradient through output gate
            tanh_c_t = self.tanh(c_t)
            do_t = dh_t * tanh_c_t
            do_raw = do_t * o_t * (1 - o_t)

            # Gradient through cell state
            dc_t = dh_t * o_t * (1 - tanh_c_t**2) + dc_next

            # Gradient through forget gate
            df_t = dc_t * c_prev
            df_raw = df_t * f_t * (1 - f_t)

            # Gradient through input gate
            di_t = dc_t * c_candidate
            di_raw = di_t * i_t * (1 - i_t)

            # Gradient through candidate values
            dc_candidate = dc_t * i_t
            dc_raw = dc_candidate * (1 - c_candidate**2)

            # Accumulate weight gradients
            dWf += np.dot(x_t.T, df_raw)
            dWi += np.dot(x_t.T, di_raw)
            dWc += np.dot(x_t.T, dc_raw)
            dWo += np.dot(x_t.T, do_raw)

            dUf += np.dot(h_prev.T, df_raw)
            dUi += np.dot(h_prev.T, di_raw)
            dUc += np.dot(h_prev.T, dc_raw)
            dUo += np.dot(h_prev.T, do_raw)

            dbf += np.sum(df_raw, axis=0)
            dbi += np.sum(di_raw, axis=0)
            dbc += np.sum(dc_raw, axis=0)
            dbo += np.sum(do_raw, axis=0)

            # Gradients for next timestep
            dh_next = (np.dot(df_raw, self.Uf.T) +
                      np.dot(di_raw, self.Ui.T) +
                      np.dot(dc_raw, self.Uc.T) +
                      np.dot(do_raw, self.Uo.T))
            dc_next = dc_t * f_t

        clip_value = 5.0

        dWf = np.clip(dWf, -clip_value, clip_value)
        dWi = np.clip(dWi, -clip_value, clip_value)
        dWc = np.clip(dWc, -clip_value, clip_value)
        dWo = np.clip(dWo, -clip_value, clip_value)

        dUf = np.clip(dUf, -clip_value, clip_value)
        dUi = np.clip(dUi, -clip_value, clip_value)
        dUc = np.clip(dUc, -clip_value, clip_value)
        dUo = np.clip(dUo, -clip_value, clip_value)

        dbf = np.clip(dbf, -clip_value, clip_value)
        dbi = np.clip(dbi, -clip_value, clip_value)
        dbc = np.clip(dbc, -clip_value, clip_value)
        dbo = np.clip(dbo, -clip_value, clip_value)

        # Update weights
        self.Wf -= learning_rate * dWf / batch_size
        self.Wi -= learning_rate * dWi / batch_size
        self.Wc -= learning_rate * dWc / batch_size
        self.Wo -= learning_rate * dWo / batch_size

        self.Uf -= learning_rate * dUf / batch_size
        self.Ui -= learning_rate * dUi / batch_size
        self.Uc -= learning_rate * dUc / batch_size
        self.Uo -= learning_rate * dUo / batch_size

        self.bf -= learning_rate * dbf / batch_size
        self.bi -= learning_rate * dbi / batch_size
        self.bc -= learning_rate * dbc / batch_size
        self.bo -= learning_rate * dbo / batch_size

        return None  
 
class DenseLayer:
    def __init__(self, input_dim, output_dim, activation='linear'):
        # Xavier initialization
        self.W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.b = np.zeros(output_dim)
        self.activation = activation

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.X = X
        self.Z = np.dot(X, self.W) + self.b

        if self.activation == 'sigmoid':
            self.A = self.sigmoid(self.Z)
        elif self.activation == 'relu':
            self.A = self.relu(self.Z)
        elif self.activation == 'softmax':
            self.A = self.softmax(self.Z)
        else:  
            self.A = self.Z

        return self.A

    def backward(self, dA, learning_rate=0.001):
        batch_size = self.X.shape[0]

        # Compute dZ based on activation function
        if self.activation == 'sigmoid':
            dZ = dA * self.A * (1 - self.A)
        elif self.activation == 'relu':
            dZ = dA * (self.Z > 0)
        elif self.activation == 'softmax':
            dZ = dA
        else: 
            dZ = dA

        # Compute gradients
        dW = np.dot(self.X.T, dZ)
        db = np.sum(dZ, axis=0)
        dX = np.dot(dZ, self.W.T)

        # Update weights
        self.W -= learning_rate * dW / batch_size
        self.b -= learning_rate * db / batch_size

        return dX
 
class LSTMModel:
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        # Create embedding matrix
        self.embedding_matrix = np.random.randn(vocab_size, embedding_dim) * 0.1

        # Initialize layers
        self.embedding = EmbeddingLayer(self.embedding_matrix)
        self.lstm = LSTMLayer(embedding_dim, hidden_dim, return_sequences=False)
        self.dense = DenseLayer(hidden_dim, num_classes, activation='linear')

    def forward(self, X):
        # X shape: (batch_size, sequence_length)
        embedded = self.embedding.forward(X)  # (batch_size, seq_len, embedding_dim)
        lstm_out = self.lstm.forward(embedded)  # (batch_size, hidden_dim)
        output = self.dense.forward(lstm_out)  # (batch_size, num_classes)
        return output

    def backward(self, dout, learning_rate=0.001):
        dout = self.dense.backward(dout, learning_rate)
        self.lstm.backward(dout, learning_rate)


class LSTM_from_Scratch:

    def __init__(self, vocab_size=None, embedding_dim=100, hidden_dim=128, num_classes=3):

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.model = None
        self.is_trained = False
        self.training_history = {'loss': [], 'accuracy': []}

    def _initialize_model(self, X):
        if self.vocab_size is None:
            self.vocab_size = np.max(X) + 1
            print(f"Auto-detected vocabulary size: {self.vocab_size}")

        self.model = LSTMModel(self.vocab_size, self.embedding_dim,
                              self.hidden_dim, self.num_classes)

    def fit(self, X_train, y_train, epochs=10, learning_rate=0.01, batch_size=32,
            validation_data=None, verbose=True):

        # Initialize model if not done
        if self.model is None:
            self._initialize_model(X_train)

        if verbose:
            print(f"Training LSTM Model")
            print(f"Data shape: X_train {X_train.shape}, y_train {y_train.shape}")
            print(f"Model parameters: vocab_size={self.vocab_size}, embedding_dim={self.embedding_dim}")
            print(f"                 hidden_dim={self.hidden_dim}, num_classes={self.num_classes}")
            print(f"Training parameters: epochs={epochs}, lr={learning_rate}, batch_size={batch_size}")
            print("="*60)

        num_samples = X_train.shape[0]

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            # Shuffle data
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            # Mini-batch training
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]

                # Forward pass
                predictions = self.model.forward(X_batch)

                # Compute loss
                loss, grad = softmax_crossentropy_loss(predictions, y_batch)
                total_loss += loss
                num_batches += 1

                # Backward pass
                self.model.backward(grad, learning_rate)

            avg_loss = total_loss / num_batches
            self.training_history['loss'].append(avg_loss)

            # Calculate training accuracy
            if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
                train_acc = self.score(X_train, y_train)
                self.training_history['accuracy'].append(train_acc)

                if verbose:
                    print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}", end="")

                    # Validation accuracy if provided
                    if validation_data is not None:
                        X_val, y_val = validation_data
                        val_acc = self.score(X_val, y_val)
                        print(f" | Val Acc: {val_acc:.4f}")
                    else:
                        print()
            elif verbose and epoch % max(1, epochs // 20) == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f}")

        self.is_trained = True
        if verbose:
            print("="*70)
            print("Training completed!")

        return self.training_history

    def predict(self, X, return_probabilities=False):
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")

        # Handle single sequence
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # Forward pass
        logits = self.model.forward(X)

        # Convert to probabilities
        probabilities = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)

        # Get predictions
        predictions = np.argmax(probabilities, axis=1)

        # Return single value for single input
        if X.shape[0] == 1:
            predictions = predictions[0]
            probabilities = probabilities[0]

        if return_probabilities:
            return predictions, probabilities
        return predictions

    def predict_proba(self, X):
        _, probabilities = self.predict(X, return_probabilities=True)
        return probabilities

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def evaluate(self, X, y, show_details=True):
        predictions, probabilities = self.predict(X, return_probabilities=True)
        accuracy = np.mean(predictions == y)

        results = {
            'accuracy': accuracy,
            'predictions': predictions,
            'probabilities': probabilities,
            'true_labels': y
        }

        if show_details:
            print(f"\nModel Evaluation Results")
            print("="*50)
            print(f"Overall Accuracy: {accuracy:.4f}")
            print(f"Total Samples: {len(y)}")

            # Per-class statistics
            print(f"\nPer-Class Results:")
            for class_id in range(self.num_classes):
                class_mask = (y == class_id)
                if np.sum(class_mask) > 0:
                    class_predictions = predictions[class_mask]
                    class_accuracy = np.mean(class_predictions == class_id)
                    class_count = np.sum(class_mask)
                    print(f"  Class {class_id}: {class_count:3d} samples, accuracy: {class_accuracy:.3f}")

            # Confidence statistics
            max_probs = np.max(probabilities, axis=1) if len(probabilities.shape) > 1 else probabilities
            print(f"\nPrediction Confidence:")
            print(f"  Mean: {np.mean(max_probs):.3f}")
            print(f"  Std:  {np.std(max_probs):.3f}")
            print(f"  Min:  {np.min(max_probs):.3f}")
            print(f"  Max:  {np.max(max_probs):.3f}")

            # Sample predictions
            print(f"\nSample Predictions (first 10):")
            print("Index | True | Pred | Confidence | Correct")
            print("-" * 42)
            for i in range(min(10, len(y))):
                true_label = y[i]
                pred_label = predictions[i]
                if len(probabilities.shape) > 1:
                    confidence = probabilities[i, pred_label]
                else:
                    confidence = probabilities[pred_label] if i == 0 else probabilities
                is_correct = "✓" if true_label == pred_label else "✗"
                print(f"{i:5d} | {true_label:4d} | {pred_label:4d} | {confidence:8.3f} | {is_correct:7s}")

        return results

    def predict_single(self, sequence, show_details=True):
        prediction, probabilities = self.predict(sequence, return_probabilities=True)

        if show_details:
            print(f"\nSingle Sequence Prediction:")
            print("="*40)
            print(f"Predicted Class: {prediction}")
            print(f"Confidence: {np.max(probabilities):.4f}")
            print(f"\nClass Probabilities:")
            for i, prob in enumerate(probabilities):
                print(f"  Class {i}: {prob:.4f}")

        return prediction, probabilities

    def get_training_history(self):
        return self.training_history

    def summary(self):
        print(f"\nLSTM Model Summary")
        print("="*40)
        print(f"Vocabulary Size: {self.vocab_size}")
        print(f"Embedding Dimension: {self.embedding_dim}")
        print(f"Hidden Dimension: {self.hidden_dim}")
        print(f"Number of Classes: {self.num_classes}")
        print(f"Model Trained: {'Yes' if self.is_trained else 'No'}")

        if self.is_trained and self.training_history['accuracy']:
            print(f"Final Training Accuracy: {self.training_history['accuracy'][-1]:.4f}")
        print("="*40)

