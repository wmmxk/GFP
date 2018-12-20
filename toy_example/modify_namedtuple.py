from collections import namedtuple


def test_modify_named_tupe():
    Config = namedtuple('Config', 'type_model, num_fold sanity')
    CONFIG = Config(type_model='svr', num_fold=10, sanity=True)

    print(CONFIG)
    print(CONFIG._replace(type_model="gp"))
    print(CONFIG)


test_modify_named_tupe()
