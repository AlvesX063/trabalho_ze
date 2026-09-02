import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


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

    # NOTAS
    ("Quero saber minha nota", "Notas"),
    ("Onde vejo minhas notas?", "Notas"),
    ("Minha nota não apareceu", "Notas"),
    ("Quando sai a nota da prova?", "Notas"),
    ("Minha nota está errada", "Notas"),
    ("Professor não lançou minha nota", "Notas"),
    ("Quero contestar minha nota", "Notas"),

    # HORÁRIOS
    ("Quero saber meu horário", "Horários"),
    ("Onde vejo o horário das aulas?", "Horários"),
    ("Qual horário da minha aula?", "Horários"),
    ("Quero consultar minha grade", "Horários"),
    ("Minha grade está errada", "Horários"),
    ("Não sei o horário da aula", "Horários"),

    # FINANCEIRO
    ("Quero saber o valor da mensalidade", "Financeiro"),
    ("Minha mensalidade está errada", "Financeiro"),
    ("Onde encontro meu boleto?", "Financeiro"),
    ("Quero negociar minha dívida", "Financeiro"),
    ("Não consigo pagar minha mensalidade", "Financeiro"),
    ("Meu boleto não apareceu", "Financeiro"),
    ("Quero segunda via do boleto", "Financeiro"),
    ("Tenho uma cobrança indevida", "Financeiro"),

    # DOCUMENTOS
    ("Preciso de uma declaração", "Documentos"),
    ("Quero meu histórico escolar", "Documentos"),
    ("Preciso de um documento da faculdade", "Documentos"),
    ("Onde pego meu histórico?", "Documentos"),
    ("Preciso de declaração de matrícula", "Documentos"),
    ("Como solicitar documentos?", "Documentos"),

    # TCC
    ("Quero saber informações sobre TCC", "TCC"),
    ("Como funciona o TCC?", "TCC"),
    ("Qual o prazo para entregar o TCC?", "TCC"),
    ("Quem pode orientar meu TCC?", "TCC"),
    ("Como escolher orientador?", "TCC"),
    ("Tenho problema com meu TCC", "TCC"),

    # PROFESSORES
    ("Quero falar com um professor", "Professores"),
    ("Preciso do contato do professor", "Professores"),
    ("Como encontro o professor?", "Professores"),
    ("Professor não respondeu", "Professores"),
    ("Quero trocar de professor", "Professores"),

    # SUPORTE DE TI
    ("Não consigo entrar no sistema", "Suporte de TI"),
    ("Meu login não funciona", "Suporte de TI"),
    ("Esqueci minha senha", "Suporte de TI"),
    ("Não consigo acessar o portal", "Suporte de TI"),
    ("O sistema está fora do ar", "Suporte de TI"),
    ("Minha senha não funciona", "Suporte de TI"),
    ("Não consigo acessar o ambiente virtual", "Suporte de TI"),

    # CANCELAMENTO
    ("Quero cancelar minha matrícula", "Cancelamento"),
    ("Quero cancelar meu curso", "Cancelamento"),
    ("Quero trancar minha faculdade", "Cancelamento"),
    ("Como faço para trancar o curso?", "Cancelamento"),
    ("Quero desistir da faculdade", "Cancelamento"),
    ("Quero cancelar minha inscrição", "Cancelamento"),

    # RECLAMAÇÃO
    ("Quero fazer uma reclamação", "Reclamação"),
    ("Estou insatisfeito com a faculdade", "Reclamação"),
    ("Quero reclamar do atendimento", "Reclamação"),
    ("Meu problema não foi resolvido", "Reclamação"),
    ("Quero reclamar de um funcionário", "Reclamação"),
    ("Estou tendo problemas com a faculdade", "Reclamação"),
]


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
            "Posso te orientar sobre matrícula.",

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
            "Posso ajudar com problemas relacionados às suas notas.",

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
            "Vamos verificar o caminho para consultar ou corrigir seu horário.",

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
            "Posso te orientar sobre boletos, mensalidades e questões financeiras.",

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
            "Posso orientar sobre a solicitação de documentos acadêmicos.",

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
            "Posso te orientar sobre os próximos passos relacionados ao TCC.",

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
            "Posso indicar o caminho para entrar em contato com um professor.",

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
            "Parece que sua solicitação está relacionada ao acesso ou funcionamento de um sistema.",

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
            "Solicitações de cancelamento ou trancamento precisam seguir as regras da instituição.",

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
# PALAVRAS QUE INDICAM NECESSIDADE DE HUMANO
# ============================================================

palavras_humano = [
    "processo judicial",
    "advogado",
    "justiça",
    "procon",
    "ameaça",
    "assédio",
    "discriminação",
    "violência",
    "fraude",
    "denúncia",
    "erro grave",
    "problema grave"
]


# ============================================================
# HISTÓRICO DO CHAT
# ============================================================

if "mensagens_chat" not in st.session_state:

    st.session_state.mensagens_chat = [

        {
            "role": "assistant",
            "content":
                "Olá! 👋 Sou o assistente acadêmico. "
                "Digite sua dúvida ou problema e vou tentar "
                "identificar o setor responsável e indicar "
                "possíveis caminhos para resolver."
        }

    ]


# ============================================================
# MOSTRAR HISTÓRICO
# ============================================================

for mensagem_chat in st.session_state.mensagens_chat:

    with st.chat_message(mensagem_chat["role"]):

        st.write(mensagem_chat["content"])


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

    # Mostrar mensagem do usuário

    st.session_state.mensagens_chat.append(
        {
            "role": "user",
            "content": mensagem
        }
    )

    with st.chat_message("user"):
        st.write(mensagem)


    # ========================================================
    # TRANSFORMAÇÃO
    # ========================================================

    mensagem_transformada = vectorizador.transform(
        [mensagem]
    )


    # ========================================================
    # CLASSIFICAÇÃO
    # ========================================================

    categoria = modelo.predict(
        mensagem_transformada
    )[0]


    probabilidades = modelo.predict_proba(
        mensagem_transformada
    )[0]


    confianca = max(probabilidades)


    setor = setores[categoria]

    dados_solucao = solucoes[categoria]


    # ========================================================
    # DETECTAR NECESSIDADE DE HUMANO
    # ========================================================

    mensagem_minuscula = mensagem.lower()

    precisa_humano = False

    for palavra in palavras_humano:

        if palavra in mensagem_minuscula:

            precisa_humano = True
            break


    if categoria in [
        "Reclamação",
        "Cancelamento"
    ]:

        precisa_humano = True


    if confianca < 0.55:

        precisa_humano = True


    # ========================================================
    # RESPOSTA DO ASSISTENTE
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
        # CAMINHO DE SOLUÇÃO
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
        # NOVA MENSAGEM
        # ----------------------------------------------------

        st.write(
            "💬 Se quiser, envie outra dúvida ou detalhe "
            "melhor o seu problema."
        )


    # Salvar resposta no histórico

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

    st.rerun()
