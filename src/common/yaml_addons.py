from copy import deepcopy
from itertools import zip_longest
import yaml 
import inspect

UNSPECIFIED = object()

class YAMLObjectFiltered(yaml.YAMLObject):
	@classmethod 
	def from_yaml(cls, loader, node):
		arg_spec = inspect.getfullargspec(cls.__init__)
		arg_spec.args.remove("self")

		arg_defaults = reversed(list(
				zip_longest(
					reversed(arg_spec.args),
					reversed(arg_spec.defaults or []),
					fillvalue=UNSPECIFIED)))

		kwarg_defaults = reversed(list(
					zip_longest(
						reversed(arg_spec.kwonlyargs),
						reversed(arg_spec.kwonlydefaults or []),
						fillvalue=UNSPECIFIED)))

		node_mapping = loader.construct_mapping(node)
		used_nodes = set()

		# fill args first
		args = []
		for a,d in arg_defaults:

			if a in node_mapping:
				args.append(node_mapping[a])
				used_nodes.add(a)
			elif d is not UNSPECIFIED:
				args.append(d)
			else:
				raise Exception(f"Tag {cls.yaml_tag} is missing '{a}' argument")
		
		# then kwargs
		kwargs = {}
		for a,d in kwarg_defaults:
			if a in node_mapping:
				kwargs[a] = node_mapping[a]
				used_nodes.add(a)
			elif d is not UNSPECIFIED:
				args[a] = d
		
		# if it accepts additional kwargs, fill with leftover kwargs
		if arg_spec.varkw and len(used_nodes) != len(node_mapping):
			for k,v in node_mapping:
				if k not in used_nodes:
					kwargs[k] = v

		return cls(*args,**kwargs)

	@classmethod
	def to_yaml(cls, dumper, data):
			"""
			Convert a Python object to a representation node.
			"""
			try:
				filtered_data = deepcopy(data)
				to_remove = set()

				for k,v in vars(filtered_data).items():
					if k not in cls.yaml_fields:
						to_remove.add(k)
				for k in to_remove:
					filtered_data.__dict__.pop(k)
			except AttributeError as E:
				raise Exception(f"YAMLObjectFiltered requires a 'yaml_fields' class field: '{cls}'. {E}")

			o = dumper.represent_yaml_object(cls.yaml_tag, filtered_data, cls,
							flow_style=cls.yaml_flow_style)
			return o

def get_loader():
	import preprocessing
	"""Add constructors to PyYAML loader."""
	loader = yaml.Loader
	return loader