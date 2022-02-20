from copy import deepcopy
from itertools import zip_longest
from typing import Tuple
import yaml 
import inspect

UNSPECIFIED = object()



def get_func_call_params_from_kwargs(func, given_kwargs) -> Tuple:
	arg_spec = inspect.getfullargspec(func)
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

	used_kwargs = set()

	# fill args first
	args = []
	for a,d in arg_defaults:
		if a in given_kwargs:
			args.append(given_kwargs[a])
			used_kwargs.add(a)
		elif d is not UNSPECIFIED:
			args.append(d)
		else:
			raise Exception(f"{func} is missing '{a}' argument")
	
	# then kwargs
	kwargs = {}
	for a,d in kwarg_defaults:
		if a in given_kwargs:
			kwargs[a] = given_kwargs[a]
			used_kwargs.add(a)
		elif d is not UNSPECIFIED:
			args[a] = d
	
	# if it accepts additional kwargs, fill with leftover kwargs
	if arg_spec.varkw and len(used_kwargs) != len(given_kwargs):
		for k,v in given_kwargs.items():
			if k not in used_kwargs:
				kwargs[k] = v

	return args,kwargs

class YAMLObjectFiltered(yaml.YAMLObject):

	def __repr__(self) -> str:
		return f"{self.__class__.__name__}: {str({k:self.__dict__.get(k,'default') for k in self.__class__.yaml_fields})}"

	@classmethod 
	def from_yaml(cls, loader, node):
		# try:
		args,kwargs = get_func_call_params_from_kwargs(cls.__init__,loader.construct_mapping(node))
		# except Exception as E:
			# raise Exception(f"When parsing Tag {cls.yaml_tag} error occured: {E}")
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

class YAMLObjectUninitializedFiltered(YAMLObjectFiltered):
	def __init__(self,**kwargs) -> None:
		for k,v in kwargs.items():
			self.__dict__[k] = v

	def create(self,**kwargs):
		target = self.__class__.yaml_class_target

		# update kwargs with values parsed from yaml,
		# combine them with kwargs passed to this function,
		for f in self.__class__.yaml_fields:
			val = self.__dict__.get(f,None)
			if val:
				kwargs[f] = val

		# figure out the order to give the arguments to the initializer in
		args,nkwargs = get_func_call_params_from_kwargs(target.__init__,kwargs)
		return target(*args,**nkwargs)

def get_loader():
	import preprocessing
	"""Add constructors to PyYAML loader."""
	loader = yaml.Loader
	return loader