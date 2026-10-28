from collections import defaultdict

from beras.core import Diffable, Tensor

class GradientTape:

    def __init__(self):
        # Dictionary mapping the object id of an output Tensor to the Diffable layer it was produced from.
        self.previous_layers: defaultdict[int, Diffable | None] = defaultdict(lambda: None)

    def __enter__(self):
        # When tape scope is entered, all Diffables will point to this tape.
        if Diffable.gradient_tape is not None:
            raise RuntimeError("Cannot nest gradient tape scopes.")

        Diffable.gradient_tape = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # When tape scope is exited, all Diffables will no longer point to this tape.
        Diffable.gradient_tape = None

    def gradient(self, target: Tensor, sources: list[Tensor]) -> list[Tensor]:
        """
        Computes the gradient of the target tensor with respect to the sources.

        :param target: the tensor to compute the gradient of, typically loss output
        :param sources: the list of tensors to compute the gradient with respect to
        In order to use tensors as keys to the dictionary, use the python built-in ID function here: https://docs.python.org/3/library/functions.html#id.
        """

        ### TODO: Populate the grads dictionary with {weight_id, weight_gradient} pairs and return the gradients of the sources

        queue = [target]                    ## Live queue; will be used to propagate backwards via breadth-first-search.
        grads = defaultdict(lambda: None)   ## Grads to be recorded. Initialize to None. Note: stores {id: list[gradients]}
        # Use id(tensor) to get the object id of a tensor object.
        # in the end, your grads dictionary should have the following structure:
        # {id(tensor): [gradient]}

        # What tensor and what gradient is for you to implement!
        # compose_input_gradients and compose_weight_gradients are methods that will be helpful

        while queue:
            current = queue.pop(0)
            layer = self.previous_layers[id(current)]

            if layer is None:
                continue

            upstream = grads[id(current)]
            input_grads = layer.compose_input_gradients(upstream)
            weight_grads = layer.compose_weight_gradients(upstream)

            for weight, grad in zip(layer.weights, weight_grads):
                if grads[id(weight)] is None:
                    grads[id(weight)] = [grad]
                else:
                    grads[id(weight)].append(grad)

            for layer_input, grad in zip(layer.inputs, input_grads):
                if grads[id(layer_input)] is None:
                    grads[id(layer_input)] = [grad]
                else:
                    grads[id(layer_input)].append(grad)

                queue.append(layer_input)
        results = []

        for source in sources:
            contributions = grads[id(source)]

            if contributions is None:
                results.append(Tensor(np.zeros_like(source)))
            else:
                results.append(Tensor(sum(contributions)))

        return results




