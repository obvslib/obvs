from tensor import Tensor
from python import Python
from python.object import PythonObject

alias TensorF32 = Tensor[DType.float32]

@value
struct SourceContext(Stringable):  # Stringable is needed for __str__ method
    """Source context for the patchscope."""

    var prompt: String
    var position: Int
    var model_name: String
    var layer: Int
    var device: String

    fn __init__(inout self, prompt: String = "", position: Int = 0, model_name: String = "gpt2", layer: Int = 0,
        device: String = "cuda:0"):

        self.prompt = prompt
        self.position = position
        self.model_name = model_name
        self.layer = layer
        self.device = device

    fn __str__(self) -> String:
        var string_rep: String = ""
        string_rep = "SourceContext(prompt=" + self.prompt + ", position=" + str(self.position) +
            ", model_name=" + self.model_name + ", layer=" + str(self.layer) + ", device=" + self.device +
            ")"

        return string_rep

# need @value decorator for assignment self.mapping_func = mapping_func in TargetContext
@value
struct IdentityFunc(Stringable):
    """Identity function for mapping in Patchscope."""
    var name: String

    fn __init__(inout self, name: String):
        self.name = name

    @staticmethod
    fn call(input: TensorF32) -> TensorF32:
        """Identity function."""
        return input

    fn __str__(self) -> String:
        return self.name

@value
struct TargetContext(Stringable):  # Stringable is needed for __str__ method
    """Target context for patchscope
    Parameters identical to the source context, with the addition of a mapping function.
    """

    var mapping_func: IdentityFunc
    var prompt: String
    var position: Int
    var model_name: String
    var layer: Int
    var device: String

    fn __init__(inout self, mapping_func: IdentityFunc, prompt: String = "", position: Int = 0, model_name: String = "gpt2", layer: Int = 0,
            device: String = "cuda:0"):

        self.mapping_func = mapping_func
        self.prompt = prompt
        self.position = position
        self.model_name = model_name
        self.layer = layer
        self.device = device

    fn __str__(self) -> String:
        var string_rep: String = ""
        string_rep = "TargetContext(prompt=" + self.prompt + ", position=" + str(self.position) +
            ", model_name=" + self.model_name + ", layer=" + str(self.layer) + ", device=" + self.device +
            ", mapping_function=" + str(self.mapping_func) + ")"

        return string_rep

struct Patchscope:

    var source_context: SourceContext
    var target_context: TargetContext

    var source_model: PythonObject
    var target_model: PythonObject


    fn __init__(inout self, source_context: SourceContext, target_context: TargetContext, source_model: PythonObject,
        target_model: PythonObject):

        self.source_context = source_context
        self.target_context = target_context
        self.source_model = source_model
        self.target_model = target_model

    fn source_forward_pass(self) raises -> PythonObject:

        # tuple unpacking is currently not supported in Mojo
        var logits_and_cache = self.source_model.run_with_cache(self.source_context.prompt)
        return logits_and_cache[1]

    fn target_forward_pass(self, source_hidden_state: PythonObject) raises -> PythonObject:

        # need to add hook function to target_model here

        #def hook_fn(target_activations, hook):
        #    target_activations[(0, self.target_context.position, :] = source_hidden_state
        #    return target_activations


        var target_logits = self.target_model.forward(self.target_context.prompt, "logits")
        return target_logits


fn main() raises:

    # create source context
    var prompt: String = "John and Mary are walking together. John said to"
    let source_context = SourceContext(
        prompt=prompt,
        model_name="gpt2",
        position=-1,  # Last token (assuming single input)
        layer=0,  # Logit lense will run on each layer.
        device="cpu"
    )

    let identity_func = IdentityFunc("Identity")

    let target_context = TargetContext(
        mapping_func=identity_func,
        prompt=prompt,
        model_name="gpt2",
        position=-1,  # Last token (assuming single input)
        layer=0,  # Logit lense will run on each layer.
        device="cpu"
    )

    print(str(source_context))
    print(str(target_context))

    # import transformer_lens
    let tl = Python.import_module("transformer_lens")
    # create source and target model
    var source_model = tl.HookedTransformer.from_pretrained(source_context.model_name)
    var target_model = tl.HookedTransformer.from_pretrained(target_context.model_name)

    # create patchscope
    let patchscope = Patchscope(
        source_context=source_context,
        target_context=target_context,
        source_model=source_model,
        target_model=target_model
    )

    # do a source model forward pass
    var source_cache = patchscope.source_forward_pass()
    var target_logits = patchscope.target_forward_pass(source_cache)
