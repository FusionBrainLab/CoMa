from typing import Any, List, Union

class AttrGetter():
    def __call__(self, obj: Any, path: List[Union[str, int]]):
        attr = obj
        for step in path:
            item_value = None
            attr_value = None
            if hasattr(attr, "__getitem__"):
                try:
                    item_value = attr[step]
                except:
                    item_value = None
            else:
                attr_value = getattr(attr, step)
            if item_value != None and attr_value != None:
                raise AttributeError(
                    "Both __getitem__ and __getattr__ returns non empty value. "
                )
            attr = item_value if item_value != None else attr_value
        return attr

class AttrSetter():
    def __call__(self, obj: Any, path: List[Union[str, int]], value: Any):
        assert len(path) >= 1, "Path must contains at least one step."
        attr_getter = AttrGetter()
        sub_obj = attr_getter(obj, path[:-1])
        is_item = True
        is_attr = True
        try:
            cur_value = sub_obj[path[-1]]
        except:
            is_item = False
        try:
            cur_value = getattr(path[-1])
        except:
            is_attr = False
        assert is_item != True or is_attr != True, f"Attrubite {path[-1]} must be settable as a dict item OR a class attribute. "
        """if is_attr:
            setattr(sub_obj, path[-1], value)
        else:
            sub_obj[path[-1]] = value"""
        if is_item == True:
            sub_obj[path[-1]] = value
        else:
            setattr(sub_obj, path[-1], value)
   