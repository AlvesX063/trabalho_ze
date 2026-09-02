import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import unicodedata
import re


# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Classificador Acadêmico",
    page_icon="🎓",
    layout="centered"
)


# ============================================================
# TÍTULO
# ============================================================

st.title("🎓 Classificador Acadêmico Inteligente")

st.write(
    "Olá! Sou o assistente virtual da faculdade. "
    "Posso identificar sua solicitação, indicar o setor responsável "
    "e apresentar possíveis caminhos para resolver seu problema."
)


# ============================================================
# BASE DE TREINAMENTO
# ============================================================

dados = [

    # MATRÍCULA
    ("Quero fazer minha matrícula", "Matrícula"),
    ("Como faço minha matrícula?", "Matrícula"),
    ("Quando começa a matrícula?", "Matrícula"),
    ("Preciso renovar minha matrícula", "Matrícula"),
    ("Não consigo fazer minha matrícula", "Matrícula"),
    ("Minha matrícula não aparece", "Matrícula"),
    ("Quero saber o prazo da matrícula", "Matrícula"),
    ("Minha matrícula está errada", "Matrícula"),
    ("Minha matrícula deu erro", "Matrícula"),
    ("Problema na matrícula", "Matrícula"),
    ("Erro na matrícula", "Matrícula"),
    ("Matrícula errada", "Matrícula"),
    ("Matrícula não funciona", "Matrícula"),

    # NOTAS
    ("Quero saber minha nota", "Notas"),
    ("Onde vejo minhas notas?", "Notas"),
    ("Minha nota não apareceu", "Notas"),
    ("Quando sai a nota da prova?", "Notas"),
    ("Minha nota está errada", "Notas"),
    ("Professor não lançou minha nota", "Notas"),
    ("Quero contestar minha nota", "Notas"),
    ("Nota errada", "Notas"),
    ("Problema com minha nota", "Notas"),
    ("Minha nota está incorreta", "Notas"),

    # HORÁRIOS
    ("Quero saber meu horário", "Horários"),
    ("Onde vejo o horário das aulas?", "Horários"),
    ("Qual horário da minha aula?", "Horários"),
    ("Quero consultar minha grade", "Horários"),
    ("Minha grade está errada", "Horários"),
    ("Não sei o horário da aula", "Horários"),
    ("Horário errado", "Horários"),
    ("Problema com meu horário", "Horários"),
    ("Minha grade está incorreta", "Horários"),

    # FINANCEIRO
    ("Quero saber o valor da mensalidade", "Financeiro"),
    ("Minha mensalidade está errada", "Financeiro"),
    ("Onde encontro meu boleto?", "Financeiro"),
    ("Quero negociar minha dívida", "Financeiro"),
    ("Não consigo pagar minha mensalidade", "Financeiro"),
    ("Meu boleto não apareceu", "Financeiro"),
    ("Quero segunda via do boleto", "Financeiro"),
    ("Tenho uma cobrança indevida", "Financeiro"),
    ("Boleto errado", "Financeiro"),
    ("Boleto vencido", "Financeiro"),
    ("Problema com boleto", "Financeiro"),
    ("Problema financeiro", "Financeiro"),
    ("Mensalidade atrasada", "Financeiro"),

    # DOCUMENTOS
    ("Preciso de uma declaração", "Documentos"),
    ("Quero meu histórico escolar", "Documentos"),
    ("Preciso de um documento da faculdade", "Documentos"),
    ("Onde pego meu histórico?", "Documentos"),
    ("Preciso de declaração de matrícula", "Documentos"),
    ("Como solicitar documentos?", "Documentos"),
    ("Preciso de um documento", "Documentos"),
    ("Problema com documento", "Documentos"),
    ("Histórico escolar", "Documentos"),

    # TCC
    ("Quero saber informações sobre TCC", "TCC"),
    ("Como funciona o TCC?", "TCC"),
    ("Qual o prazo para entregar o TCC?", "TCC"),
    ("Quem pode orientar meu TCC?", "TCC"),
    ("Como escolher orientador?", "TCC"),
    ("Tenho problema com meu TCC", "TCC"),
    ("Problema no TCC", "TCC"),
    ("Orientação do TCC", "TCC"),

    # PROFESSORES
    ("Quero falar com um professor", "Professores"),
    ("Preciso do contato do professor", "Professores"),
    ("Como encontro o professor?", "Professores"),
    ("Professor não respondeu", "Professores"),
    ("Quero trocar de professor", "Professores"),
    ("Problema com professor", "Professores"),
    ("Meu professor não responde", "Professores"),

    # SUPORTE DE TI
    ("Não consigo entrar no sistema", "Suporte de TI"),
    ("Meu login não funciona", "Suporte de TI"),
    ("Esqueci minha senha", "Suporte de TI"),
    ("Não consigo acessar o portal", "Suporte de TI"),
    ("O sistema está fora do ar", "Suporte de TI"),
    ("Minha senha não funciona", "Suporte de TI"),
    ("Não consigo acessar o ambiente virtual", "Suporte de TI"),
    ("Problema no sistema", "Suporte de TI"),
    ("Erro no sistema", "Suporte de TI"),
    ("Portal não funciona", "Suporte de TI"),
    ("Não consigo entrar", "Suporte de TI"),

    # CANCELAMENTO
    ("Quero cancelar minha matrícula", "Cancelamento"),
    ("Quero cancelar meu curso", "Cancelamento"),
    ("Quero trancar minha faculdade", "Cancelamento"),
    ("Como faço para trancar o curso?", "Cancelamento"),
    ("Quero desistir da faculdade", "Cancelamento"),
    ("Quero cancelar minha inscrição", "Cancelamento"),
    ("Quero cancelar", "Cancelamento"),
    ("Quero trancar", "Cancelamento"),
    ("Quero desistir", "Cancelamento"),

    # RECLAMAÇÃO
    ("Quero fazer uma reclamação", "Reclamação"),
    ("Estou insatisfeito com a faculdade", "Reclamação"),
    ("Quero reclamar do atendimento", "Reclamação"),
    ("Meu problema não foi resolvido", "Reclamação"),
    ("Quero reclamar de um funcionário", "Reclamação"),
    ("Estou tendo problemas com a faculdade", "Reclamação"),
    ("Quero reclamar", "Reclamação"),
    ("Tenho uma reclamação", "Reclamação"),
]


# ============================================================
# FUNÇÃO PARA NORMALIZAR TEXTO
# ============================================================

def normalizar_texto(texto):

    texto = texto.lower()

    texto = unicodedata.normalize(
        "NFD",
        texto
    )

    texto = "".join(
        caractere
        for caractere in texto
        if unicodedata.category(caractere) != "Mn"
    )

    texto = re.sub(
        r"[^a-z0-9\s]",
        " ",
        texto
    )

    texto = re.sub(
        r"\s+",
        " ",
        texto
    ).strip()

    return texto


# ============================================================
# PREPARAÇÃO DO MODELO
# ============================================================

mensagens = [item[0] for item in dados]
categorias = [item[1] for item in dados]

vectorizador = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2)
)

X = vectorizador.fit_transform(mensagens)

modelo = LogisticRegression(
    max_iter=1000
)

modelo.fit(X, categorias)


# ============================================================
# SETORES
# ============================================================

setores = {

    "Matrícula":
        "Secretaria Acadêmica",

    "Notas":
        "Secretaria Acadêmica / Coordenação",

    "Horários":
        "Secretaria Acadêmica",

    "Financeiro":
        "Setor Financeiro",

    "Documentos":
        "Secretaria Acadêmica",

    "TCC":
        "Coordenação do Curso",

    "Professores":
        "Coordenação do Curso",

    "Suporte de TI":
        "Suporte de Tecnologia da Informação",

    "Cancelamento":
        "Secretaria Acadêmica / Atendimento",

    "Reclamação":
        "Atendimento ao Aluno"
}


# ============================================================
# SOLUÇÕES E CAMINHOS
# ============================================================

solucoes = {

    "Matrícula": {

        "resposta":
            "Entendi! Sua solicitação está relacionada à matrícula.",

        "passos": [
            "Acesse o portal acadêmico.",
            "Procure a opção 'Matrícula' ou 'Renovação de Matrícula'.",
            "Confira as disciplinas disponíveis.",
            "Confirme seus dados e finalize a solicitação.",
            "Verifique se o sistema gerou um comprovante."
        ],

        "documentos":
            "Normalmente podem ser solicitados documentos pessoais e comprovantes acadêmicos.",

        "humano":
            "Se o sistema impedir a matrícula, houver divergência de dados ou o prazo tiver encerrado, procure a Secretaria Acadêmica."
    },


    "Notas": {

        "resposta":
            "Entendi! Sua solicitação está relacionada às notas.",

        "passos": [
            "Acesse o portal do aluno.",
            "Entre na área de notas ou desempenho acadêmico.",
            "Confira se a disciplina e o período estão corretos.",
            "Caso a nota ainda não esteja disponível, aguarde o prazo informado pela instituição.",
            "Se o prazo já tiver passado, entre em contato com a Secretaria ou Coordenação."
        ],

        "documentos":
            "Tenha em mãos seu nome, matrícula, disciplina e informações da avaliação.",

        "humano":
            "Contestação de nota ou alteração de nota exige análise de um responsável acadêmico."
    },


    "Horários": {

        "resposta":
            "Entendi! Sua solicitação está relacionada aos horários ou à grade.",

        "passos": [
            "Acesse o portal acadêmico.",
            "Procure por 'Grade Horária' ou 'Horários'.",
            "Confira as disciplinas e os horários.",
            "Verifique também a sala e o professor.",
            "Caso exista divergência, informe a Secretaria Acadêmica."
        ],

        "documentos":
            "Tenha em mãos seu curso, período e turma.",

        "humano":
            "Alterações de turma, disciplina ou horário precisam ser analisadas pela instituição."
    },


    "Financeiro": {

        "resposta":
            "Entendi! Sua solicitação está relacionada à parte financeira.",

        "passos": [
            "Acesse o portal financeiro da instituição.",
            "Procure pela área de pagamentos ou boletos.",
            "Confira se existem parcelas em aberto.",
            "Caso o boleto não esteja disponível, tente gerar uma segunda via.",
            "Se houver cobrança incorreta, entre em contato com o Setor Financeiro."
        ],

        "documentos":
            "Tenha em mãos seus dados acadêmicos e informações da cobrança.",

        "humano":
            "Negociação de dívida, contestação de cobrança e alterações financeiras precisam de atendimento humano."
    },


    "Documentos": {

        "resposta":
            "Entendi! Sua solicitação está relacionada a documentos acadêmicos.",

        "passos": [
            "Acesse o portal acadêmico.",
            "Procure por 'Solicitação de Documentos'.",
            "Escolha o documento desejado.",
            "Preencha os dados solicitados.",
            "Acompanhe o prazo de emissão."
        ],

        "documentos":
            "Pode ser necessário informar matrícula, curso e tipo de documento.",

        "humano":
            "Se o documento não estiver disponível no portal ou houver erro nos dados, procure a Secretaria Acadêmica."
    },


    "TCC": {

        "resposta":
            "Entendi! Sua solicitação está relacionada ao TCC.",

        "passos": [
            "Consulte as regras do TCC do seu curso.",
            "Verifique o calendário e os prazos.",
            "Confirme quem são os professores disponíveis para orientação.",
            "Escolha o tema conforme as regras do curso.",
            "Mantenha contato com seu orientador durante o desenvolvimento."
        ],

        "documentos":
            "Consulte as normas e documentos oficiais disponibilizados pela instituição.",

        "humano":
            "Escolha de orientador, alteração de prazo e problemas com avaliação devem ser tratados com a Coordenação."
    },


    "Professores": {

        "resposta":
            "Entendi! Sua solicitação está relacionada aos professores.",

        "passos": [
            "Verifique o portal acadêmico.",
            "Procure a área de professores ou contatos.",
            "Confira se existe e-mail institucional.",
            "Caso não encontre o contato, procure a Coordenação do Curso."
        ],

        "documentos":
            "Tenha em mãos o nome do professor e a disciplina.",

        "humano":
            "Problemas relacionados a conduta, ausência ou conflitos com professores devem ser analisados pela Coordenação."
    },


    "Suporte de TI": {

        "resposta":
            "Entendi! Sua solicitação parece estar relacionada ao acesso ou funcionamento de um sistema.",

        "passos": [
            "Confira se sua internet está funcionando.",
            "Tente atualizar a página.",
            "Tente acessar utilizando outro navegador.",
            "Confira se seu login e senha estão corretos.",
            "Se esqueceu a senha, utilize a opção 'Esqueci minha senha'.",
            "Se o problema continuar, procure o Suporte de TI."
        ],

        "documentos":
            "Informe seu usuário, sistema afetado e uma descrição do erro. Nunca informe sua senha.",

        "humano":
            "Se o problema continuar após essas tentativas, o Suporte de TI deverá analisar o caso."
    },


    "Cancelamento": {

        "resposta":
            "Entendi. Sua solicitação está relacionada a cancelamento, trancamento ou desistência.",

        "passos": [
            "Consulte as regras acadêmicas do curso.",
            "Verifique se o pedido pode ser realizado pelo portal.",
            "Confira possíveis prazos e pendências.",
            "Faça a solicitação pelo canal oficial da instituição.",
            "Guarde o protocolo ou comprovante."
        ],

        "documentos":
            "Podem ser solicitados documentos pessoais e informações acadêmicas.",

        "humano":
            "⚠️ Cancelamento, trancamento e desistência exigem atendimento humano."
    },


    "Reclamação": {

        "resposta":
            "Entendo. Sua situação pode precisar de uma análise mais detalhada.",

        "passos": [
            "Reúna as informações sobre o problema.",
            "Anote datas, horários e nomes envolvidos.",
            "Guarde protocolos e comprovantes de atendimento.",
            "Registre a reclamação pelo canal oficial da instituição.",
            "Solicite e guarde o número de protocolo."
        ],

        "documentos":
            "Tenha em mãos protocolos, comprovantes, mensagens ou outros registros relacionados ao problema.",

        "humano":
            "⚠️ Sua situação deve ser analisada por um atendente. O assistente virtual não substitui uma análise humana."
    }
}


# ============================================================
# PALAVRAS-CHAVE POR CATEGORIA
# ============================================================

palavras_chave_categoria = {

    "Matrícula": [
        "matricula",
        "rematricula",
        "renovar matricula",
        "renovacao de matricula",
        "inscricao"
    ],

    "Notas": [
        "nota",
        "notas",
        "prova",
        "resultado",
        "avaliacao",
        "lancamento de nota"
    ],

    "Horários": [
        "horario",
        "horarios",
        "grade",
        "grade horaria",
        "aula",
        "sala",
        "turma"
    ],

    "Financeiro": [
        "boleto",
        "mensalidade",
        "mensalidades",
        "cobranca",
        "divida",
        "pagamento",
        "pagar",
        "financeiro",
        "parcela",
        "parcelas",
        "juros"
    ],

    "Documentos": [
        "documento",
        "documentos",
        "historico",
        "declaracao",
        "certificado",
        "comprovante",
        "atestado"
    ],

    "TCC": [
        "tcc",
        "trabalho de conclusao",
        "orientador",
        "orientacao do tcc"
    ],

    "Professores": [
        "professor",
        "professora",
        "professores",
        "docente",
        "docentes"
    ],

    "Suporte de TI": [
        "login",
        "senha",
        "acesso",
        "portal",
        "sistema",
        "site",
        "ambiente virtual",
        "erro no sistema",
        "nao consigo entrar",
        "nao consigo acessar"
    ],

    "Cancelamento": [
        "cancelar",
        "cancelamento",
        "trancar",
        "trancamento",
        "desistir",
        "desistencia"
    ],

    "Reclamação": [
        "reclamacao",
        "reclamar",
        "insatisfeito",
        "denuncia",
        "problema grave"
    ]
}


# ============================================================
# PALAVRAS QUE INDICAM NECESSIDADE DE HUMANO
# ============================================================

palavras_humano = [
    "processo judicial",
    "advogado",
    "justica",
    "procon",
    "ameaca",
    "assedio",
    "discriminacao",
    "violencia",
    "fraude",
    "denuncia",
    "erro grave",
    "problema grave"
]


# ============================================================
# NOVO - FRASES QUE INDICAM QUE A SOLUÇÃO NÃO FUNCIONOU
# ============================================================

palavras_problema_nao_resolvido = [

    "nao funcionou",
    "não funcionou",

    "nao resolveu",
    "não resolveu",

    "ainda nao funciona",
    "ainda não funciona",

    "ainda esta dando erro",
    "ainda está dando erro",

    "continua dando erro",

    "continua com problema",

    "problema continua",

    "nao deu certo",
    "não deu certo",

    "continua igual",

    "continua acontecendo",

    "nao consegui resolver",
    "não consegui resolver",

    "nao consigo resolver",
    "não consigo resolver",

    "mesmo assim nao funciona",
    "mesmo assim não funciona",

    "ja tentei",
    "já tentei",

    "tentei e nao funcionou",
    "tentei e não funcionou",

    "tentei e nao resolveu",
    "tentei e não resolveu"
]


# ============================================================
# SAUDAÇÕES
# ============================================================

saudacoes = [
    "oi",
    "ola",
    "oie",
    "oii",
    "oiii",
    "bom dia",
    "boa tarde",
    "boa noite",
    "tudo bem",
    "tudo bom",
    "como voce esta",
    "como vc esta",
    "quem e voce"
]


# ============================================================
# PERGUNTAS DE CONTINUAÇÃO
# ============================================================

palavras_continuacao = [

    "e como",
    "como faco",
    "e agora",
    "o que faco",
    "onde vejo",
    "onde encontro",
    "onde faco",
    "qual o prazo",
    "e se",

    "nao funcionou",
    "não funcionou",

    "nao deu certo",
    "não deu certo",

    "ainda nao",
    "ainda não",

    "entendi",
    "pode explicar",
    "explique melhor",
    "mais informacoes",
    "como resolvo",
    "como resolver",
    "o que devo fazer",
    "qual caminho"
]


# ============================================================
# PEDIDOS DE ATENDIMENTO HUMANO
# ============================================================

palavras_pedir_humano = [

    "quero falar com alguem",
    "quero atendente",
    "quero falar com atendente",
    "preciso de um atendente",
    "quero atendimento humano",
    "atendimento humano",
    "falar com uma pessoa",
    "falar com alguem",
    "nao quero falar com robo",
    "quero falar com uma pessoa"
]


# ============================================================
# FUNÇÃO - SAUDAÇÃO
# ============================================================

def eh_saudacao(texto):

    texto = normalizar_texto(texto)

    if texto in saudacoes:
        return True

    return False


# ============================================================
# FUNÇÃO - CONTINUAÇÃO
# ============================================================

def eh_continuacao(texto):

    texto = normalizar_texto(texto)

    for palavra in palavras_continuacao:

        if normalizar_texto(palavra) in texto:

            return True

    return False


# ============================================================
# NOVA FUNÇÃO - PROBLEMA NÃO RESOLVIDO
# ============================================================

def eh_problema_nao_resolvido(texto):

    texto_normalizado = normalizar_texto(texto)

    for frase in palavras_problema_nao_resolvido:

        if normalizar_texto(frase) in texto_normalizado:

            return True

    return False


# ============================================================
# FUNÇÃO - ENCONTRAR CATEGORIA POR PALAVRAS-CHAVE
# ============================================================

def encontrar_categoria_por_palavras(texto):

    texto = normalizar_texto(texto)

    pontuacao = {}

    for categoria, palavras in palavras_chave_categoria.items():

        pontuacao[categoria] = 0

        for palavra in palavras:

            palavra_normalizada = normalizar_texto(palavra)

            if palavra_normalizada in texto:

                quantidade_palavras = len(
                    palavra_normalizada.split()
                )

                if quantidade_palavras >= 2:

                    pontuacao[categoria] += 3

                else:

                    pontuacao[categoria] += 2


    if max(pontuacao.values()) == 0:

        return None


    melhor_categoria = max(
        pontuacao,
        key=pontuacao.get
    )


    maiores = [

        categoria

        for categoria, pontos in pontuacao.items()

        if pontos == pontuacao[melhor_categoria]

    ]


    if len(maiores) > 1:

        return None


    return melhor_categoria


# ============================================================
# RESPOSTA DE SAUDAÇÃO
# ============================================================

def responder_saudacao():

    return (
        "Olá! 👋😊 Seja bem-vindo ao atendimento acadêmico.\n\n"

        "Posso te ajudar a identificar o caminho mais adequado "
        "para resolver sua solicitação.\n\n"

        "Você pode perguntar sobre:\n\n"

        "🎓 Matrícula\n"
        "📝 Notas\n"
        "🕐 Horários\n"
        "💰 Financeiro\n"
        "📄 Documentos\n"
        "📚 TCC\n"
        "👨‍🏫 Professores\n"
        "💻 Acesso ao sistema\n"
        "❌ Cancelamento\n"
        "📢 Reclamações\n\n"

        "Pode escrever sua dúvida do jeito que você falaria "
        "com um atendente."
    )


# ============================================================
# RESPOSTA DE CONTINUAÇÃO
# ============================================================

def responder_continuacao(categoria):

    dados = solucoes[categoria]

    resposta = (

        f"Claro! 👍 Como estamos falando sobre **{categoria}**, "
        f"vou continuar te orientando por esse caminho.\n\n"

    )

    resposta += dados["resposta"] + "\n\n"

    resposta += "🛠️ **Caminho recomendado:**\n\n"


    for i, passo in enumerate(
        dados["passos"],
        start=1
    ):

        resposta += f"**{i}.** {passo}\n\n"


    resposta += (

        f"🏢 **Setor responsável:** {setores[categoria]}\n\n"

        f"📋 **Informações que podem ser necessárias:** "
        f"{dados['documentos']}\n\n"

    )


    resposta += (

        "💬 Se você já tentou essas orientações e "
        "o problema não foi resolvido, me diga o que aconteceu.\n\n"

        "👨‍💼 Caso continue sem solução, recomendo procurar "
        "o atendimento humano da instituição para uma análise individual."
    )


    return resposta


# ============================================================
# NOVA RESPOSTA - SOLUÇÃO NÃO FUNCIONOU
# ============================================================

def responder_solucao_nao_funcionou():

    return (

        "😕 Entendi. Se você já tentou as orientações que passei "
        "e **o problema continua**, provavelmente é uma situação "
        "que precisa ser analisada individualmente.\n\n"

        "💡 Para não ficar repetindo as mesmas orientações, "
        "o melhor caminho agora é procurar o atendimento humano "
        "da instituição.\n\n"

        "👨‍💼 **Recomendo entrar em contato com um atendente.**\n\n"

        "O atendente poderá analisar seu caso diretamente, "
        "verificar seus dados e indicar o procedimento correto.\n\n"

        "📋 Ao entrar em contato, se possível, informe:\n\n"

        "• Seu nome e matrícula;\n"
        "• Curso e período;\n"
        "• O que aconteceu;\n"
        "• O que você já tentou fazer;\n"
        "• Qual mensagem de erro apareceu, se houver;\n"
        "• Protocolos ou comprovantes relacionados ao problema.\n\n"

        "⚠️ **Nunca informe sua senha ou códigos de acesso.**"
    )


# ============================================================
# RESPOSTA PARA ATENDIMENTO HUMANO
# ============================================================

def responder_humano():

    return (

        "👨‍💼 **Claro. Nesse caso, o melhor caminho é o atendimento humano.**\n\n"

        "O assistente virtual consegue fornecer orientações iniciais, "
        "mas algumas situações precisam ser analisadas individualmente "
        "por um funcionário da instituição.\n\n"

        "📋 Recomendo informar ao atendente:\n\n"

        "• Seu nome e matrícula;\n"
        "• O curso e período;\n"
        "• O que aconteceu;\n"
        "• Quando o problema ocorreu;\n"
        "• Protocolos ou comprovantes, se houver.\n\n"

        "⚠️ Nunca informe sua senha ou códigos de acesso.\n\n"

        "Se quiser, também posso tentar identificar primeiro "
        "qual setor deve receber sua solicitação."
    )


# ============================================================
# RESPOSTA QUANDO NÃO ENTENDEU
# ============================================================

def responder_nao_entendi():

    return (

        "🤔 **Quero te ajudar, mas ainda não consegui identificar "
        "com segurança qual é o seu problema.**\n\n"

        "Tente me explicar um pouco mais, por exemplo:\n\n"

        "• O que aconteceu?\n"
        "• Qual sistema ou serviço você está tentando utilizar?\n"
        "• O que você precisa resolver?\n"
        "• Apareceu alguma mensagem de erro?\n\n"

        "💡 Se você já tentou as orientações anteriores e "
        "**elas não resolveram o problema**, não precisa ficar "
        "tentando por aqui.\n\n"

        "👨‍💼 **Nesse caso, recomendo entrar em contato com "
        "o atendimento humano da instituição.**\n\n"

        "Um atendente poderá analisar sua situação individualmente "
        "e orientar você sobre o procedimento correto."
    )


# ============================================================
# HISTÓRICO DO CHAT
# ============================================================

if "mensagens_chat" not in st.session_state:

    st.session_state.mensagens_chat = [

        {
            "role": "assistant",

            "content":
                "Olá! 👋 Sou o assistente acadêmico. "
                "Como posso ajudar? Você pode escrever sua dúvida "
                "normalmente, como faria com um atendente."
        }

    ]


# ============================================================
# MEMÓRIA DA ÚLTIMA CATEGORIA
# ============================================================

if "ultima_categoria" not in st.session_state:

    st.session_state.ultima_categoria = None


# ============================================================
# MOSTRAR HISTÓRICO
# ============================================================

for mensagem_chat in st.session_state.mensagens_chat:

    with st.chat_message(
        mensagem_chat["role"]
    ):

        st.write(
            mensagem_chat["content"]
        )


# ============================================================
# CAMPO DE CHAT
# ============================================================

mensagem = st.chat_input(
    "Digite sua dúvida ou solicitação..."
)


# ============================================================
# PROCESSAMENTO
# ============================================================

if mensagem:

    # ========================================================
    # MOSTRAR MENSAGEM DO USUÁRIO
    # ========================================================

    st.session_state.mensagens_chat.append(
        {
            "role": "user",
            "content": mensagem
        }
    )


    with st.chat_message("user"):

        st.write(mensagem)


    mensagem_normalizada = normalizar_texto(
        mensagem
    )


    # ========================================================
    # 1 - PEDIDO DE ATENDIMENTO HUMANO
    # ========================================================

    quer_humano = False


    for palavra in palavras_pedir_humano:

        palavra_normalizada = normalizar_texto(
            palavra
        )

        if palavra_normalizada in mensagem_normalizada:

            quer_humano = True

            break


    if quer_humano:

        resposta_chat = responder_humano()


        with st.chat_message("assistant"):

            st.write(resposta_chat)


        st.session_state.mensagens_chat.append(
            {
                "role": "assistant",
                "content": resposta_chat
            }
        )


        st.stop()


    # ========================================================
    # 2 - VERIFICAR SE A SOLUÇÃO ANTERIOR NÃO FUNCIONOU
    # ========================================================

    if eh_problema_nao_resolvido(mensagem):

        resposta_chat = responder_solucao_nao_funcionou()


        with st.chat_message("assistant"):

            st.write(resposta_chat)


            st.error(
                "👨‍💼 **Atendimento humano recomendado:** "
                "o problema continua mesmo após as tentativas."
            )


        st.session_state.mensagens_chat.append(
            {
                "role": "assistant",
                "content": resposta_chat
            }
        )


        st.stop()


    # ========================================================
    # 3 - SAUDAÇÃO
    # ========================================================

    if eh_saudacao(mensagem):

        resposta_chat = responder_saudacao()


        with st.chat_message("assistant"):

            st.write(resposta_chat)


        st.session_state.mensagens_chat.append(
            {
                "role": "assistant",
                "content": resposta_chat
            }
        )


        st.stop()


    # ========================================================
    # 4 - ENCONTRAR CATEGORIA POR PALAVRAS-CHAVE
    # ========================================================

    categoria_por_palavra = encontrar_categoria_por_palavras(
        mensagem
    )


    # ========================================================
    # 5 - CONTINUAÇÃO DA CONVERSA
    # ========================================================

    if (

        st.session_state.ultima_categoria is not None

        and categoria_por_palavra is None

        and eh_continuacao(mensagem)

    ):

        categoria = st.session_state.ultima_categoria


        resposta_chat = responder_continuacao(
            categoria
        )


        with st.chat_message("assistant"):

            st.write(resposta_chat)


        st.session_state.mensagens_chat.append(
            {
                "role": "assistant",
                "content": resposta_chat
            }
        )


        st.stop()


    # ========================================================
    # 6 - CLASSIFICAÇÃO DO MODELO
    # ========================================================

    mensagem_transformada = vectorizador.transform(
        [mensagem]
    )


    categoria_modelo = modelo.predict(
        mensagem_transformada
    )[0]


    probabilidades = modelo.predict_proba(
        mensagem_transformada
    )[0]


    confianca_modelo = max(
        probabilidades
    )


    # ========================================================
    # 7 - ESCOLHER MELHOR CLASSIFICAÇÃO
    # ========================================================

    if categoria_por_palavra is not None:

        categoria = categoria_por_palavra

        confianca = max(
            confianca_modelo,
            0.85
        )

    else:

        categoria = categoria_modelo

        confianca = confianca_modelo


    # ========================================================
    # 8 - SE NÃO HOUVER SEGURANÇA
    # ========================================================

    if (

        categoria_por_palavra is None

        and confianca < 0.55

    ):

        resposta_chat = responder_nao_entendi()


        with st.chat_message("assistant"):

            st.write(resposta_chat)


            st.error(
                "👨‍💼 **Se o problema continuar sem solução, "
                "procure um atendente da instituição.**"
            )


        st.session_state.mensagens_chat.append(
            {
                "role": "assistant",
                "content": resposta_chat
            }
        )


        st.stop()


    # ========================================================
    # 9 - SALVAR MEMÓRIA
    # ========================================================

    st.session_state.ultima_categoria = categoria


    # ========================================================
    # 10 - SETOR E SOLUÇÃO
    # ========================================================

    setor = setores[categoria]

    dados_solucao = solucoes[categoria]


    # ========================================================
    # 11 - DETECTAR NECESSIDADE DE HUMANO
    # ========================================================

    precisa_humano = False


    for palavra in palavras_humano:

        palavra_normalizada = normalizar_texto(
            palavra
        )

        if palavra_normalizada in mensagem_normalizada:

            precisa_humano = True

            break


    if categoria in [

        "Reclamação",
        "Cancelamento"

    ]:

        precisa_humano = True


    # ========================================================
    # 12 - RESPOSTA DO ASSISTENTE
    # ========================================================

    with st.chat_message("assistant"):

        st.write(
            dados_solucao["resposta"]
        )


        st.divider()


        # ----------------------------------------------------
        # CATEGORIA
        # ----------------------------------------------------

        st.write(
            f"🏷️ **Categoria identificada:** {categoria}"
        )


        # ----------------------------------------------------
        # SETOR
        # ----------------------------------------------------

        st.write(
            f"🏢 **Setor responsável:** {setor}"
        )


        # ----------------------------------------------------
        # CONFIANÇA
        # ----------------------------------------------------

        st.write(
            f"🤖 **Confiança da classificação:** "
            f"{confianca * 100:.1f}%"
        )


        st.progress(
            float(confianca)
        )


        # ----------------------------------------------------
        # CAMINHO
        # ----------------------------------------------------

        st.subheader(
            "🛠️ Possíveis caminhos para resolver"
        )


        for i, passo in enumerate(

            dados_solucao["passos"],

            start=1

        ):

            st.write(
                f"**{i}.** {passo}"
            )


        # ----------------------------------------------------
        # DOCUMENTOS
        # ----------------------------------------------------

        st.subheader(
            "📋 Informações que podem ser necessárias"
        )


        st.info(
            dados_solucao["documentos"]
        )


        # ----------------------------------------------------
        # ATENDIMENTO HUMANO
        # ----------------------------------------------------

        if precisa_humano:

            st.error(
                "👨‍💼 **Atendimento humano recomendado**"
            )


            st.write(
                dados_solucao["humano"]
            )


            st.warning(
                "⚠️ O assistente virtual fornece uma orientação "
                "inicial, mas não substitui o atendimento "
                "da instituição."
            )

        else:

            st.success(
                "✅ Você pode tentar seguir os passos acima "
                "antes de procurar atendimento humano."
            )


        # ----------------------------------------------------
        # CONTINUAÇÃO
        # ----------------------------------------------------

        st.write(
            "💬 Continue a conversa se quiser. "
            "Posso usar o assunto anterior para interpretar "
            "sua próxima dúvida."
        )


    # ========================================================
    # SALVAR RESUMO
    # ========================================================

    resumo_resposta = (

        f"Categoria: {categoria}\n\n"

        f"Setor: {setor}\n\n"

        f"Confiança: {confianca * 100:.1f}%\n\n"

        f"Atendimento humano: "
        f"{'Sim' if precisa_humano else 'Não'}"

    )


    st.session_state.mensagens_chat.append(
        {
            "role": "assistant",
            "content": resumo_resposta
        }
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🎓 Sobre o sistema")


    st.write(
        "Sistema de classificação automática de "
        "solicitações acadêmicas."
    )


    st.divider()


    st.subheader("📌 Categorias")


    for categoria_nome in setores:

        st.write(
            f"• {categoria_nome}"
        )


    st.divider()


    st.caption(
        "Projeto acadêmico — Sistema de Informações Acadêmicas"
    )


# ============================================================
# BOTÃO LIMPAR CONVERSA
# ============================================================

if st.sidebar.button(
    "🗑️ Limpar conversa"
):

    st.session_state.mensagens_chat = [

        {
            "role": "assistant",

            "content":
                "Olá! 👋 Sou o assistente acadêmico. "
                "Como posso ajudar?"
        }

    ]


    st.session_state.ultima_categoria = None


    st.rerun()
