import argparse

def init(args, config):
    for key, value in config.items():
        if type(value) == dict:
            if key == 'bind_key':
                for _, group in value.items():
                    key1 = group['key']
                    value1 = group['value']
                    for name in key1:
                        current_namespace = args
                        namespace = name.split('.')
                        for layer in namespace[:-1]:
                            if not hasattr(current_namespace, layer):
                                setattr(current_namespace, layer, argparse.Namespace())
                            current_namespace = getattr(current_namespace, layer)
                        setattr(current_namespace, namespace[-1], value1)
            elif key == 'add_key':
                setattr(args, key, argparse.Namespace())
                current_namespace = getattr(args, key)
                for key1, value1 in value.items():
                    setattr(current_namespace, key1, value1)
            else:
                setattr(args, key, init(argparse.Namespace(), value))
        else:
            setattr(args, key, value)
    return args

def add_key(args, add_label, value_used):
    command_list = []
    
    if not hasattr(args.add_key, add_label):
        print(f"Config don't have add_key of {add_label}, please check")
        exit(0)
    for key, value in value_used.items():
        exec(f'{key} = {value}')
    dic = getattr(args.add_key, add_label)
    for _, group in dic.items():
        add_list = group['key']
        add_value = group['value']
        for value_name in add_list:
            current_namespace = args
            namespace = value_name.split('.')
            for layer in namespace[:-1]:
                current_namespace = getattr(current_namespace, layer)
            command_list.append(f'setattr(current_namespace, "{namespace[-1]}", {add_value})')
            exec(f'setattr(current_namespace, "{namespace[-1]}", {add_value})')
    return args