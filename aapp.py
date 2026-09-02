import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Classificador Acadêmico",
    page_icon="🎓"
)

st.title("🎓 Classificador Acadêmico Inteligente")

st.write(
    "Digite uma solicitação acadêmica para que o sistema "
    "identifique a categoria e o setor responsável."
)

mensagem = st.text_area(
    "Digite sua solicitação:",
    placeholder="Exemplo: Quero saber quando começa a matrícula..."
)

dados = [
    ("Quero fazer minha matrícula", "Matrícula"),
    ("Como faço minha matrícula?", "Matrícula"),
    ("Quando começa a matrícula?", "Matrícula"),
    ("Preciso renovar minha matrícula", "Matrícula"),

    ("Quero saber minha nota", "Notas"),
    ("Onde vejo minhas notas?", "Notas"),
    ("Minha nota não apareceu", "Notas"),
    ("Quando sai a nota da prova?", "Notas"),

    ("Quero saber meu horário", "Horários"),
    ("Onde vejo o horário das aulas?", "Horários"),
    ("Qual horário da minha aula?", "Horários"),
    ("Quero consultar minha grade", "Horários"),

    ("Quero saber o valor da mensalidade", "Financeiro"),
    ("Minha mensalidade está errada", "Financeiro"),
    ("Onde encontro meu boleto?", "Financeiro"),
    ("Quero negociar minha dívida", "Financeiro"),

    ("Preciso de uma declaração", "Documentos"),
    ("Quero meu histórico escolar", "Documentos"),
    ("Preciso de um documento da faculdade", "Documentos"),
    ("Onde pego meu histórico?", "Documentos"),

    ("Quero saber informações sobre TCC", "TCC"),
    ("Como funciona o TCC?", "TCC"),
    ("Qual o prazo para entregar o TCC?", "TCC"),
    ("Quem pode orientar meu TCC?", "TCC"),

    ("Quero falar com um professor", "Professores"),
    ("Preciso do contato do professor", "Professores"),
    ("Como encontro o professor?", "Professores"),

    ("Não consigo entrar no sistema", "Suporte de TI"),
    ("Meu login não funciona", "Suporte de TI"),
    ("Esqueci minha senha", "Suporte de TI"),
    ("Não consigo acessar o portal", "Suporte de TI"),

    ("Quero cancelar minha matrícula", "Cancelamento"),
    ("Quero cancelar meu curso", "Cancelamento"),
    ("Quero trancar minha faculdade", "Cancelamento"),
    ("Como faço para trancar o curso?", "Cancelamento"),

    ("Quero fazer uma reclamação", "Reclamação"),
    ("Estou insatisfeito com a faculdade", "Reclamação"),
    ("Quero reclamar do atendimento", "Reclamação"),
    ("Meu problema não foi resolvido", "Reclamação")
]

mensagens = [item[0] for item in dados]
categorias = [item[1] for item in dados]

vectorizador = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode"
)

X = vectorizador.fit_transform(mensagens)

modelo = LogisticRegression(max_iter=1000)

modelo.fit(X, categorias)

setores = {
    "Matrícula": "Secretaria Acadêmica",
    "Notas": "Secretaria Acadêmica / Coordenação",
    "Horários": "Secretaria Acadêmica",
    "Financeiro": "Setor Financeiro",
    "Documentos": "Secretaria Acadêmica",
    "TCC": "Coordenação do Curso",
    "Professores": "Coordenação do Curso",
    "Suporte de TI": "Suporte de Tecnologia da Informação",
    "Cancelamento": "Secretaria Acadêmica / Atendimento",
    "Reclamação": "Atendimento ao Aluno"
}

respostas = {
    "Matrícula":
        "Olá! Para informações sobre matrícula, "
        "consulte o calendário acadêmico ou procure a Secretaria Acadêmica.",

    "Notas":
        "Olá! Você pode consultar suas notas pelo portal do aluno. "
        "Caso exista algum problema, procure a Secretaria ou Coordenação.",

    "Horários":
        "Olá! Os horários das disciplinas podem ser consultados "
        "no portal acadêmico ou com a Secretaria Acadêmica.",

    "Financeiro":
        "Olá! Para informações sobre mensalidades, boletos "
        "ou pendências, entre em contato com o Setor Financeiro.",

    "Documentos":
        "Olá! Documentos acadêmicos podem ser solicitados "
        "à Secretaria Acadêmica.",

    "TCC":
        "Olá! Para informações sobre TCC, prazos e orientadores, "
        "entre em contato com a Coordenação do Curso.",

    "Professores":
        "Olá! Para entrar em contato com professores, "
        "procure a Coordenação do Curso.",

    "Suporte de TI":
        "Olá! Parece que existe um problema relacionado ao sistema. "
        "Entre em contato com o Suporte de Tecnologia da Informação.",

    "Cancelamento":
        "Olá! Solicitações de cancelamento ou trancamento "
        "precisam ser analisadas pela instituição.",

    "Reclamação":
        "Olá! Sentimos muito pelo ocorrido. "
        "Sua solicitação será encaminhada ao Atendimento ao Aluno."
    }

if st.button("🔍 Analisar solicitação"):

    if mensagem.strip() == "":
        st.warning("Digite uma solicitação antes de analisar.")

    else:
        mensagem_transformada = vectorizador.transform([mensagem])

        categoria = modelo.predict(mensagem_transformada)[0]

        probabilidades = modelo.predict_proba(
            mensagem_transformada
        )[0]

        confianca = max(probabilidades)

        setor = setores[categoria]

        resposta = respostas[categoria]

        st.divider()

        st.subheader("📊 Resultado")

        st.write("🏷️ **Categoria:**", categoria)

        st.write("🏢 **Setor responsável:**", setor)

        st.write(
            f"🤖 **Confiança do modelo:** "
            f"{confianca * 100:.1f}%"
        )

        st.progress(float(confianca))

        st.subheader("💬 Resposta sugerida")

        st.info(resposta)

        if categoria in ["Reclamação", "Cancelamento"] or confianca < 0.55:

            st.error(
                "⚠️ Atendimento humano recomendado."
            )

        else:

            st.success(
                "✅ A solicitação pode receber uma resposta automática inicial."
            )