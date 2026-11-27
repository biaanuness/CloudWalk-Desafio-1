"""Parâmetros da análise, todos declarados ANTES de olhar o conjunto de teste."""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Configuracao:
    semente: int = 20261117              # mesma semente dos materiais do desafio
    threads: int = 2                     # mesmo número de threads do treinamento
    # Escolha da referência (somente validação)
    n_candidatos: tuple[int, ...] = (25, 50, 100, 200, 400)
    n_ouro: int = 2000                   # referência grande usada como "gabarito"
    sementes_validacao: int = 5
    linhas_validacao: int = 200
    limite_erro_rel: float = 0.10        # regra: menor N com erro <= 10% ...
    limite_jaccard: float = 0.80         # ... e Jaccard top-3 >= 0,80
    # Casos do teste: percentis da probabilidade prevista
    quantis_casos: dict[str, float] = field(
        default_factory=lambda: {"baixo": 0.10, "intermediario": 0.50, "alto": 0.90})
    # Fidelidade
    linhas_fidelidade: int = 300
    k_max: int = 5
    sorteios_aleatorios: int = 5
    # Estabilidade
    sementes_referencia: int = 20
    perturbacoes: int = 50
    # Tempo
    repeticoes_tempo: int = 3

    def rapida(self) -> "Configuracao":
        """Versão reduzida para demonstração (~15 s). Os números finais do relatório
        vêm do modo completo; o modo rápido serve para ver o fluxo de ponta a ponta."""
        return replace(self, n_candidatos=(25, 100, 200), n_ouro=500, sementes_validacao=2,
                       linhas_validacao=50, linhas_fidelidade=30, sorteios_aleatorios=2,
                       sementes_referencia=5, perturbacoes=10)
