class Jogador:

    nome: str = "messi"

    def __init__(self, nome: str) -> None:
        self.nome = nome

    @classmethod
    def get_class_nome(cls):
        return cls.nome

    def get_self_nome(self):
        return self.nome


print(Jogador.nome)
print(Jogador("MARCOS").nome)
