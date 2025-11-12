import torch.nn as nn

# Relative imports for local AttnLRP implementation
from .rules import WrapModule
from .modules import INIT_MODULE_MAPPING
from .check import WHITELIST, BLACKLIST, SYMBOLS

# Optional FX tracer (only needed for function-level replacements)
try:  # transformers may be unavailable
    from transformers.utils.fx import HFTracer, get_concrete_args  # type: ignore
    from torch.fx import GraphModule
except Exception:  # pragma: no cover
    HFTracer = None  # type: ignore
    get_concrete_args = None  # type: ignore
    GraphModule = None  # type: ignore


class Composite:
    """Register LRP/AttnLRP rules on a model.

    - Module-level replacement: swap nn.Modules with rule-wrapped modules
    - Function-level replacement (optional): via FX if tracer + dummy_inputs are provided
    """

    def __init__(self, layer_map, canonizers=None, zennit_composite=None) -> None:
        self.layer_map = layer_map or {}
        self.original_modules = []
        self.module_summary, self.function_summary = {}, {}

        self.canonizers = canonizers or []
        self.canonizer_instances = []
        for c in self.canonizers:
            if isinstance(c, type):
                raise ValueError("Canonizer must be an instance (call Canonizer()), not a class.")

        self.zennit_composite = zennit_composite

    def register(self, parent: nn.Module, dummy_inputs: dict = None, tracer=None, verbose: bool = False, no_grad: bool = True):
        if no_grad:
            for p in parent.parameters():
                p.requires_grad = False

        # Apply canonizers (optional)
        for canonizer in self.canonizers:
            try:
                instances = canonizer.apply(parent, verbose)
            except TypeError:
                instances = canonizer.apply(parent)
            self.canonizer_instances.extend(instances)

        module_map, fn_map = self._parse_rules(self.layer_map)
        if module_map:
            self._iterate_children(parent, module_map)

        # Function-level replacement only if tracer and dummy_inputs are provided and available
        if (fn_map or dummy_inputs) and HFTracer is not None and GraphModule is not None and get_concrete_args is not None:
            used_tracer = tracer or HFTracer
            parent = self._iterate_graph(parent, dummy_inputs or {}, fn_map, module_map, used_tracer)

        if self.zennit_composite:
            if verbose:
                print("-> register Zennit composite", self.zennit_composite)
            self.zennit_composite.register(parent)

        if verbose and (fn_map or dummy_inputs):
            self.print_summary()

        return parent

    def _parse_rules(self, layer_map):
        module_map, fn_map = {}, {}
        for key, value in layer_map.items():
            if isinstance(key, str) or isinstance(key, type):
                module_map[key] = value
            elif callable(key):
                fn_map[key] = value
            else:
                raise ValueError("layer_map keys must be nn.Module subclass, string, or callable.")
        return module_map, fn_map

    def _iterate_children(self, parent: nn.Module, rule_dict):
        for name, child in parent.named_children():
            child = self._attach_module_rule(child, parent, name, rule_dict)
            self._iterate_children(child, rule_dict)

    def _attach_module_rule(self, child, parent, name, rule_dict):
        for layer_type, rule in rule_dict.items():
            matched = False
            if isinstance(layer_type, str):
                matched = (child.__class__.__name__ == layer_type)
            elif isinstance(layer_type, type):
                matched = isinstance(child, layer_type)
            if not matched:
                continue

            if isinstance(rule, type) and issubclass(rule, WrapModule):
                xai_module = rule(child)
                setattr(parent, name, xai_module)
                self.original_modules.append((parent, name, child))
                return xai_module
            elif isinstance(rule, type) and issubclass(rule, nn.Module):
                xai_module = INIT_MODULE_MAPPING[rule](child, rule)
                setattr(parent, name, xai_module)
                self.original_modules.append((parent, name, child))
                return xai_module
            else:
                raise ValueError("rule must be a WrapModule subclass or nn.Module subclass")
        return child

    def _iterate_graph(self, model, dummy_inputs, fn_map, module_map, tracer):
        assert isinstance(dummy_inputs, dict) and dummy_inputs, "dummy_inputs must be a non-empty dict"
        graph = tracer().trace(model, concrete_args=get_concrete_args(model, dummy_inputs.keys()), dummy_inputs=dummy_inputs)

        module_types = list(module_map.values())
        for node in graph.nodes:
            self._attach_function_rule(node, fn_map, module_types)

        graph.lint()
        traced = GraphModule(model, graph)
        traced.recompile()
        return traced

    def _attach_function_rule(self, node, fn_map, module_types):
        if self._check_already_wrapped(node, module_types):
            self._add_to_module_summary(node, True)
            return False

        if node.op == 'call_function':
            if node.target in fn_map:
                self._add_to_fn_summary(node, True)
                node.target = fn_map[node.target]
                return True
            self._add_to_fn_summary(node, False)
        elif node.op == 'call_method':
            self._add_to_fn_summary(node, False)
        elif node.op == 'call_module':
            self._add_to_module_summary(node, False)
        return False

    def _check_already_wrapped(self, node, module_types):
        if "nn_module_stack" in node.meta:
            for _, l_type in node.meta["nn_module_stack"].items():
                if l_type in module_types:
                    return True
        return False

    def _add_to_module_summary(self, node, replaced: bool):
        if "nn_module_stack" not in node.meta:
            return
        _, l_type = list(node.meta["nn_module_stack"].items())[-1]
        if l_type not in self.module_summary:
            self.module_summary[l_type] = replaced

    def _add_to_fn_summary(self, node, replaced: bool):
        if "nn_module_stack" in node.meta:
            module_name = list(node.meta["nn_module_stack"].values())[-1]
        else:
            module_name = "Root"
        if module_name not in self.function_summary:
            self.function_summary[module_name] = {}
        if replaced:
            self.function_summary[module_name][node.target] = "replaced"
        elif node.target in WHITELIST:
            self.function_summary[module_name][node.target] = "compatible"
        elif node.target in BLACKLIST:
            self.function_summary[module_name][node.target] = "problematic"
        else:
            self.function_summary[module_name][node.target] = "unknown"

    def print_summary(self):
        headers = ["Module", "Function", "Replaced", "LRP-Compatible"]
        data = []
        for module in self.module_summary:
            if self.module_summary[module]:
                replaced = SYMBOLS.get("true", "Y")
                compatible = "-"
            else:
                replaced = "-"
                compatible = SYMBOLS.get("unknown", "?")
            data.append([module, "-", replaced, compatible])

