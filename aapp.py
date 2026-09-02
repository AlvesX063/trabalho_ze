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
# NOVO - BASE DE CONHECIMENTO ESPECÍFICO
# ============================================================

conhecimento_especifico = {

    "Matrícula": {

        "calendario": {
            "palavras": [
                "calendario",
                "calendario academico",
                "calendario escolar",
                "datas importantes",
                "datas da faculdade",
                "prazo da faculdade",
                "prazos da faculdade"
            ],

            "resposta": (
                "📅 **Calendário Acadêmico**\n\n"
                "O calendário acadêmico é o documento que apresenta "
                "as principais datas e prazos da instituição durante "
                "o período letivo.\n\n"

                "📌 Normalmente ele apresenta:\n\n"
                "• Período de matrícula;\n"
                "• Início e término das aulas;\n"
                "• Período de provas e avaliações;\n"
                "• Prazos para solicitações acadêmicas;\n"
                "• Férias e recessos;\n"
                "• Datas importantes do semestre.\n\n"

                "🔎 **Como consultar:**\n\n"
                "1. Acesse o portal acadêmico da instituição.\n"
                "2. Procure por **Calendário Acadêmico**, "
                "**Calendário Escolar** ou **Datas Importantes**.\n"
                "3. Selecione o período letivo atual.\n"
                "4. Confira a data relacionada à sua solicitação.\n\n"

                "🏢 Caso não encontre o calendário ou tenha dúvida "
                "sobre algum prazo, procure a **Secretaria Acadêmica**."
            )
        },

        "como fazer": {
            "palavras": [
                "como fazer matricula",
                "como faco matricula",
                "fazer matricula",
                "realizar matricula",
                "efetuar matricula"
            ],

            "resposta": (
                "🎓 **Como fazer a matrícula**\n\n"
                "A matrícula é o procedimento utilizado para confirmar "
                "o vínculo do aluno com a instituição e, normalmente, "
                "selecionar ou confirmar as disciplinas do período.\n\n"

                "🛠️ **Como realizar:**\n\n"
                "1. Acesse o portal acadêmico.\n"
                "2. Procure a opção **Matrícula** ou "
                "**Renovação de Matrícula**.\n"
                "3. Confira as disciplinas disponíveis.\n"
                "4. Verifique seus dados acadêmicos.\n"
                "5. Confirme a matrícula.\n"
                "6. Guarde o comprovante.\n\n"

                "⚠️ Se a opção não estiver disponível ou aparecer "
                "algum erro, procure a **Secretaria Acadêmica**."
            )
        },

        "prazo": {
            "palavras": [
                "prazo matricula",
                "prazo da matricula",
                "quando termina matricula",
                "quando acaba matricula",
                "quando comeca matricula",
                "quando começa matricula"
            ],

            "resposta": (
                "📅 **Prazo da matrícula**\n\n"
                "O prazo de matrícula é definido pela instituição e "
                "normalmente aparece no calendário acadêmico.\n\n"

                "🔎 Para verificar:\n"
                "1. Consulte o Calendário Acadêmico.\n"
                "2. Procure a data referente à matrícula.\n"
                "3. Verifique se o período ainda está aberto.\n\n"

                "⚠️ Se o prazo já tiver encerrado, procure a "
                "**Secretaria Acadêmica** para verificar se existe "
                "algum procedimento disponível."
            )
        },

        "erro": {
            "palavras": [
                "erro matricula",
                "erro na matricula",
                "matricula deu erro",
                "matricula nao funciona",
                "matricula não funciona"
            ],

            "resposta": (
                "⚠️ **Erro ao realizar a matrícula**\n\n"
                "Um erro na matrícula pode ocorrer por diferentes "
                "motivos, como pendências acadêmicas, problemas "
                "cadastrais, prazo encerrado ou instabilidade do sistema.\n\n"

                "🔎 Primeiro confira:\n"
                "• Se o prazo de matrícula está aberto;\n"
                "• Se existem pendências financeiras ou acadêmicas;\n"
                "• Se seus dados estão corretos;\n"
                "• Se o portal está funcionando normalmente.\n\n"

                "Se o erro continuar, procure a **Secretaria Acadêmica** "
                "e informe a mensagem apresentada pelo sistema."
            )
        }
    },


    "Notas": {

        "onde ver": {
            "palavras": [
                "onde vejo nota",
                "onde vejo minhas notas",
                "onde consultar nota",
                "como ver nota",
                "consultar minhas notas"
            ],

            "resposta": (
                "📝 **Onde consultar suas notas**\n\n"
                "As notas representam os resultados obtidos pelo aluno "
                "nas avaliações realizadas durante o período letivo.\n\n"

                "🔎 **Para consultar:**\n"
                "1. Acesse o portal acadêmico.\n"
                "2. Entre em **Notas**, **Boletim** ou "
                "**Desempenho Acadêmico**.\n"
                "3. Selecione o período letivo.\n"
                "4. Escolha a disciplina desejada.\n\n"

                "Se a nota não aparecer dentro do prazo previsto, "
                "procure a **Secretaria Acadêmica ou Coordenação**."
            )
        },

        "nota nao apareceu": {
            "palavras": [
                "nota nao apareceu",
                "nota não apareceu",
                "nota nao foi lancada",
                "nota não foi lançada",
                "professor nao lancou",
                "professor não lançou"
            ],

            "resposta": (
                "⏳ **Minha nota não apareceu**\n\n"
                "Quando uma nota não aparece no sistema, pode ser que "
                "o lançamento ainda não tenha sido realizado ou que "
                "o sistema ainda não tenha atualizado a informação.\n\n"

                "📌 Verifique:\n"
                "• Se a disciplina está correta;\n"
                "• Se o período letivo está correto;\n"
                "• Se o prazo de lançamento já terminou.\n\n"

                "Se o prazo já passou e a nota continua ausente, "
                "procure a **Coordenação ou Secretaria Acadêmica**."
            )
        },

        "nota errada": {
            "palavras": [
                "nota errada",
                "nota incorreta",
                "nota esta errada",
                "minha nota esta errada",
                "contestar nota",
                "contestacao de nota"
            ],

            "resposta": (
                "⚠️ **Nota incorreta ou contestação de nota**\n\n"
                "Se você acredita que uma nota foi lançada "
                "incorretamente, é importante verificar a avaliação "
                "e os critérios utilizados.\n\n"

                "📌 Tenha em mãos:\n"
                "• Disciplina;\n"
                "• Avaliação;\n"
                "• Nota apresentada;\n"
                "• Informações que justifiquem a contestação.\n\n"

                "A correção ou alteração da nota precisa ser analisada "
                "pelo professor ou responsável acadêmico."
            )
        },

        "media": {
            "palavras": [
                "media",
                "media final",
                "calcular media",
                "calculo da media",
                "nota minima",
                "media para passar"
            ],

            "resposta": (
                "📊 **Média acadêmica**\n\n"
                "A média acadêmica é o resultado utilizado pela "
                "instituição para avaliar o desempenho do aluno "
                "em determinada disciplina.\n\n"

                "⚠️ A forma de cálculo pode variar conforme o curso "
                "e as regras da instituição.\n\n"

                "Por isso, consulte o regulamento acadêmico ou o "
                "professor da disciplina para confirmar os pesos, "
                "avaliações e média mínima exigida."
            )
        }
    },


    "Horários": {

        "onde ver": {
            "palavras": [
                "onde vejo horario",
                "onde vejo horarios",
                "como ver horario",
                "consultar horario",
                "ver meu horario"
            ],

            "resposta": (
                "🕐 **Consulta de horários**\n\n"
                "O horário ou grade horária apresenta os dias e "
                "horários das disciplinas que você deverá cursar.\n\n"

                "🔎 Para consultar:\n"
                "1. Acesse o portal acadêmico.\n"
                "2. Procure por **Horários** ou **Grade Horária**.\n"
                "3. Confira as disciplinas e horários.\n"
                "4. Verifique também sala, turma e professor."
            )
        },

        "grade": {
            "palavras": [
                "grade",
                "grade horaria",
                "minha grade",
                "grade de aulas",
                "grade das aulas"
            ],

            "resposta": (
                "📚 **Grade Horária**\n\n"
                "A grade horária organiza as disciplinas do aluno "
                "de acordo com os dias e horários das aulas.\n\n"

                "Ela pode apresentar:\n"
                "• Disciplina;\n"
                "• Dia da semana;\n"
                "• Horário;\n"
                "• Sala;\n"
                "• Professor;\n"
                "• Turma.\n\n"

                "Consulte o portal acadêmico para visualizar sua grade."
            )
        },

        "sala": {
            "palavras": [
                "qual sala",
                "onde e minha sala",
                "onde fica minha sala",
                "sala da aula",
                "sala da disciplina"
            ],

            "resposta": (
                "🏫 **Sala da aula**\n\n"
                "A sala indica o local onde determinada disciplina "
                "será realizada presencialmente.\n\n"

                "🔎 Verifique a grade horária ou o portal acadêmico "
                "para encontrar a sala vinculada à disciplina.\n\n"

                "Se a sala não aparecer ou estiver incorreta, "
                "procure a **Secretaria Acadêmica**."
            )
        },

        "horario errado": {
            "palavras": [
                "horario errado",
                "horario incorreto",
                "grade errada",
                "grade incorreta",
                "horario mudou",
                "mudaram meu horario"
            ],

            "resposta": (
                "⚠️ **Problema com o horário**\n\n"
                "Se sua grade apresentar um horário diferente do "
                "esperado, confira primeiro se houve alguma alteração "
                "recente na turma ou disciplina.\n\n"

                "Se a informação continuar incorreta, procure a "
                "**Secretaria Acadêmica**."
            )
        }
    },


    "Financeiro": {

        "boleto": {
            "palavras": [
                "onde encontro boleto",
                "onde vejo boleto",
                "como pegar boleto",
                "como gerar boleto",
                "meu boleto"
            ],

            "resposta": (
                "💰 **Boleto da mensalidade**\n\n"
                "O boleto é o documento utilizado para realizar o "
                "pagamento de uma mensalidade ou outra cobrança.\n\n"

                "🔎 Normalmente você pode encontrá-lo no portal "
                "financeiro ou na área de pagamentos.\n\n"

                "1. Acesse o portal.\n"
                "2. Entre em **Financeiro** ou **Pagamentos**.\n"
                "3. Localize a parcela.\n"
                "4. Gere ou visualize o boleto.\n\n"

                "Se ele não estiver disponível, procure o "
                "**Setor Financeiro**."
            )
        },

        "segunda via": {
            "palavras": [
                "segunda via",
                "segunda via boleto",
                "gerar segunda via",
                "boleto novamente"
            ],

            "resposta": (
                "🧾 **Segunda via do boleto**\n\n"
                "A segunda via permite gerar novamente um boleto "
                "quando o documento original não está disponível.\n\n"

                "🔎 Acesse o portal financeiro, localize a parcela "
                "e procure pela opção **Segunda Via** ou "
                "**Gerar Boleto**.\n\n"

                "Se a opção não estiver disponível, procure o "
                "**Setor Financeiro**."
            )
        },

        "mensalidade": {
            "palavras": [
                "valor mensalidade",
                "valor da mensalidade",
                "quanto custa mensalidade",
                "preco da mensalidade",
                "mensalidade"
            ],

            "resposta": (
                "💵 **Mensalidade**\n\n"
                "A mensalidade é o valor cobrado pela instituição "
                "pela prestação dos serviços educacionais.\n\n"

                "O valor pode variar conforme curso, período, "
                "bolsas, descontos ou condições contratuais.\n\n"

                "Para consultar o valor exato, verifique o portal "
                "financeiro ou procure o **Setor Financeiro**."
            )
        },

        "divida": {
            "palavras": [
                "divida",
                "divida atrasada",
                "mensalidade atrasada",
                "negociar divida",
                "negociacao",
                "parcelar divida"
            ],

            "resposta": (
                "💰 **Pendência financeira**\n\n"
                "Uma pendência financeira ocorre quando existe uma "
                "cobrança em aberto ou não paga dentro do prazo.\n\n"

                "Para verificar a situação, consulte o portal "
                "financeiro e identifique as parcelas pendentes.\n\n"

                "⚠️ Negociação, parcelamento e condições de pagamento "
                "devem ser analisados pelo **Setor Financeiro**."
            )
        }
    },


    "Documentos": {

        "historico": {
            "palavras": [
                "historico",
                "historico escolar",
                "onde pego historico",
                "como pegar historico"
            ],

            "resposta": (
                "📄 **Histórico Escolar**\n\n"
                "O histórico escolar é um documento acadêmico que "
                "registra informações da trajetória do aluno, como "
                "disciplinas cursadas, resultados e períodos acadêmicos.\n\n"

                "🔎 Para solicitar:\n"
                "1. Acesse o portal acadêmico.\n"
                "2. Procure por **Documentos** ou "
                "**Solicitação de Documentos**.\n"
                "3. Localize o histórico escolar.\n"
                "4. Solicite o documento.\n"
                "5. Acompanhe o prazo de emissão.\n\n"

                "Se não estiver disponível, procure a "
                "**Secretaria Acadêmica**."
            )
        },

        "declaracao": {
            "palavras": [
                "declaracao",
                "declaracao de matricula",
                "declaracao escolar",
                "declaracao de aluno"
            ],

            "resposta": (
                "📋 **Declaração acadêmica**\n\n"
                "A declaração é um documento emitido pela instituição "
                "para comprovar determinada informação acadêmica do aluno.\n\n"

                "Ela pode ser utilizada, por exemplo, para comprovar "
                "matrícula ou vínculo acadêmico, conforme o documento "
                "disponibilizado pela instituição.\n\n"

                "🔎 Normalmente a solicitação pode ser realizada "
                "pelo portal acadêmico, na área de documentos."
            )
        },

        "prazo": {
            "palavras": [
                "prazo documento",
                "prazo do documento",
                "quanto demora documento",
                "quando fica pronto documento"
            ],

            "resposta": (
                "⏳ **Prazo para emissão de documentos**\n\n"
                "O prazo é o período necessário para que a instituição "
                "processe e disponibilize o documento solicitado.\n\n"

                "O prazo pode variar conforme o tipo de documento "
                "e as regras da instituição.\n\n"

                "🔎 Consulte a solicitação realizada no portal "
                "acadêmico para acompanhar o andamento."
            )
        }
    },


    "TCC": {

        "conceito": {
            "palavras": [
                "o que e tcc",
                "o que é tcc",
                "tcc o que e",
                "tcc o que é"
            ],

            "resposta": (
                "📚 **O que é o TCC?**\n\n"
                "TCC significa **Trabalho de Conclusão de Curso**. "
                "É uma atividade acadêmica desenvolvida pelo aluno "
                "como parte da conclusão da graduação, conforme as "
                "regras do curso.\n\n"

                "O trabalho pode envolver pesquisa, análise de um "
                "problema, desenvolvimento de um projeto ou outro "
                "formato definido pela instituição.\n\n"

                "📌 Normalmente o aluno precisa:\n"
                "• Definir um tema;\n"
                "• Ter um orientador;\n"
                "• Desenvolver o trabalho;\n"
                "• Seguir as normas acadêmicas;\n"
                "• Entregar o trabalho dentro do prazo;\n"
                "• Apresentar o trabalho, quando exigido."
            )
        },

        "prazo": {
            "palavras": [
                "prazo tcc",
                "prazo do tcc",
                "entrega tcc",
                "quando entregar tcc",
                "data entrega tcc"
            ],

            "resposta": (
                "📅 **Prazo de entrega do TCC**\n\n"
                "O prazo de entrega é definido pelo curso e normalmente "
                "aparece no calendário acadêmico ou no cronograma "
                "específico do TCC.\n\n"

                "🔎 Consulte:\n"
                "• Calendário acadêmico;\n"
                "• Cronograma do TCC;\n"
                "• Orientações da Coordenação;\n"
                "• Informações fornecidas pelo orientador.\n\n"

                "⚠️ Se você não encontrar a data oficial, procure "
                "a **Coordenação do Curso**."
            )
        },

        "orientador": {
            "palavras": [
                "quem pode orientar",
                "como escolher orientador",
                "escolher orientador",
                "orientador tcc",
                "professor orientador"
            ],

            "resposta": (
                "👨‍🏫 **Orientador do TCC**\n\n"
                "O orientador é o professor responsável por acompanhar "
                "o desenvolvimento acadêmico do TCC e auxiliar o aluno "
                "durante a elaboração do trabalho.\n\n"

                "A escolha normalmente depende das regras do curso, "
                "da disponibilidade dos professores e da área de "
                "interesse do aluno.\n\n"

                "Para saber quais professores podem orientar seu trabalho, "
                "procure a **Coordenação do Curso**."
            )
        }
    },


    "Professores": {

        "contato": {
            "palavras": [
                "contato professor",
                "contato do professor",
                "email professor",
                "telefone professor",
                "como falar com professor"
            ],

            "resposta": (
                "👨‍🏫 **Contato do professor**\n\n"
                "O contato do professor normalmente pode ser encontrado "
                "no portal acadêmico, ambiente virtual de aprendizagem "
                "ou por meio do e-mail institucional.\n\n"

                "🔎 Procure:\n"
                "1. Portal acadêmico;\n"
                "2. Área de professores;\n"
                "3. Ambiente virtual;\n"
                "4. E-mail institucional.\n\n"

                "Se não encontrar o contato, procure a "
                "**Coordenação do Curso**."
            )
        },

        "nao responde": {
            "palavras": [
                "professor nao responde",
                "professor não responde",
                "professor nao respondeu",
                "professor não respondeu"
            ],

            "resposta": (
                "📩 **Professor não respondeu**\n\n"
                "Se o professor ainda não respondeu, confira primeiro "
                "se a mensagem foi enviada pelo canal correto e se "
                "houve tempo suficiente para o retorno.\n\n"

                "Se o contato continuar sem resposta e a situação "
                "estiver prejudicando sua atividade acadêmica, "
                "procure a **Coordenação do Curso**.\n\n"

                "📋 Se possível, guarde o registro da mensagem enviada."
            )
        },

        "troca": {
            "palavras": [
                "trocar professor",
                "troca de professor",
                "mudar professor",
                "mudanca de professor"
            ],

            "resposta": (
                "🔄 **Troca de professor**\n\n"
                "A troca de professor depende das regras da instituição "
                "e da existência de justificativa ou disponibilidade.\n\n"

                "Esse tipo de solicitação normalmente precisa ser "
                "avaliado pela **Coordenação do Curso**.\n\n"

                "Explique o motivo da solicitação e informe a disciplina "
                "envolvida."
            )
        }
    },


    "Suporte de TI": {

        "senha": {
            "palavras": [
                "esqueci minha senha",
                "esqueci senha",
                "recuperar senha",
                "resetar senha",
                "redefinir senha",
                "senha esquecida"
            ],

            "resposta": (
                "🔐 **Esqueci minha senha**\n\n"
                "A senha é utilizada para autenticar seu acesso aos "
                "sistemas acadêmicos.\n\n"

                "🔎 Para recuperar:\n"
                "1. Acesse a tela de login.\n"
                "2. Procure **Esqueci minha senha** ou "
                "**Recuperar senha**.\n"
                "3. Informe os dados solicitados.\n"
                "4. Siga as instruções recebidas.\n\n"

                "⚠️ Nunca informe sua senha para outra pessoa."
            )
        },

        "login": {
            "palavras": [
                "login",
                "nao consigo entrar",
                "não consigo entrar",
                "entrar no sistema",
                "acessar sistema"
            ],

            "resposta": (
                "💻 **Problema de login**\n\n"
                "O login é o processo utilizado para acessar o sistema "
                "com seu usuário e senha.\n\n"

                "Se não conseguir entrar:\n"
                "1. Confira seu usuário;\n"
                "2. Confira sua senha;\n"
                "3. Verifique sua conexão com a internet;\n"
                "4. Tente outro navegador;\n"
                "5. Utilize a opção de recuperação de senha, se necessário.\n\n"

                "Se continuar sem acesso, procure o **Suporte de TI**."
            )
        },

        "portal": {
            "palavras": [
                "portal",
                "portal academico",
                "portal nao funciona",
                "portal não funciona"
            ],

            "resposta": (
                "🌐 **Portal acadêmico**\n\n"
                "O portal acadêmico é o ambiente utilizado pelo aluno "
                "para acessar serviços e informações da instituição, "
                "como notas, matrícula, documentos e horários.\n\n"

                "Se o portal não funcionar:\n"
                "• Atualize a página;\n"
                "• Teste outro navegador;\n"
                "• Verifique sua internet;\n"
                "• Tente novamente após alguns minutos.\n\n"

                "Se o problema persistir, procure o **Suporte de TI**."
            )
        },

        "erro": {
            "palavras": [
                "erro sistema",
                "erro no sistema",
                "sistema com erro",
                "problema sistema",
                "sistema nao funciona",
                "sistema não funciona"
            ],

            "resposta": (
                "⚠️ **Erro no sistema**\n\n"
                "Um erro no sistema pode estar relacionado ao acesso, "
                "navegador, conexão ou ao próprio serviço da instituição.\n\n"

                "🔎 Tente:\n"
                "1. Atualizar a página;\n"
                "2. Fechar e abrir novamente;\n"
                "3. Utilizar outro navegador;\n"
                "4. Verificar sua internet.\n\n"

                "Se o erro continuar, informe ao **Suporte de TI** "
                "qual mensagem apareceu na tela.\n\n"

                "⚠️ Nunca envie sua senha."
            )
        }
    },


    "Cancelamento": {

        "cancelamento": {
            "palavras": [
                "cancelar matricula",
                "cancelar curso",
                "cancelamento matricula",
                "cancelamento curso"
            ],

            "resposta": (
                "❌ **Cancelamento**\n\n"
                "O cancelamento é o procedimento utilizado para "
                "encerrar o vínculo acadêmico conforme as regras "
                "da instituição.\n\n"

                "Antes de solicitar, verifique:\n"
                "• Prazos;\n"
                "• Pendências financeiras;\n"
                "• Consequências acadêmicas;\n"
                "• Regras do contrato ou instituição.\n\n"

                "⚠️ Como o cancelamento pode gerar consequências "
                "acadêmicas e financeiras, procure a "
                "**Secretaria Acadêmica / Atendimento**."
            )
        },

        "trancamento": {
            "palavras": [
                "trancar",
                "trancamento",
                "trancar faculdade",
                "trancar curso"
            ],

            "resposta": (
                "⏸️ **Trancamento**\n\n"
                "O trancamento normalmente permite interromper "
                "temporariamente os estudos mantendo determinadas "
                "condições de vínculo com a instituição.\n\n"

                "As regras, prazos e condições variam conforme "
                "a instituição e o curso.\n\n"

                "🔎 Antes de solicitar, verifique as regras "
                "acadêmicas e procure a **Secretaria Acadêmica**."
            )
        },

        "desistencia": {
            "palavras": [
                "desistir",
                "desistencia",
                "desistir da faculdade",
                "desistir do curso"
            ],

            "resposta": (
                "⚠️ **Desistência do curso**\n\n"
                "A desistência representa a decisão do aluno de "
                "não continuar o curso, mas o procedimento correto "
                "depende das regras da instituição.\n\n"

                "Antes de realizar a solicitação, confirme as "
                "consequências acadêmicas e financeiras.\n\n"

                "🏢 Recomendo procurar a **Secretaria Acadêmica / "
                "Atendimento** para receber a orientação correta."
            )
        }
    },


    "Reclamação": {

        "como reclamar": {
            "palavras": [
                "como fazer reclamacao",
                "como faco reclamacao",
                "onde reclamar",
                "registrar reclamacao",
                "fazer reclamacao"
            ],

            "resposta": (
                "📢 **Como registrar uma reclamação**\n\n"
                "A reclamação é utilizada para registrar formalmente "
                "uma insatisfação relacionada a um serviço, atendimento "
                "ou situação ocorrida na instituição.\n\n"

                "📌 Ao registrar:\n"
                "1. Explique claramente o que aconteceu;\n"
                "2. Informe data e local, quando aplicável;\n"
                "3. Identifique o setor ou pessoa envolvida, se souber;\n"
                "4. Anexe comprovantes quando necessário;\n"
                "5. Solicite um número de protocolo.\n\n"

                "O registro deve ser realizado pelo canal oficial "
                "de atendimento da instituição."
            )
        },

        "protocolo": {
            "palavras": [
                "protocolo",
                "numero de protocolo",
                "numero do protocolo",
                "acompanhar reclamacao"
            ],

            "resposta": (
                "📋 **Protocolo de atendimento**\n\n"
                "O protocolo é um número ou registro utilizado para "
                "identificar e acompanhar uma solicitação ou reclamação.\n\n"

                "Guarde o número do protocolo após registrar sua "
                "solicitação.\n\n"

                "Ele pode ser utilizado para consultar o andamento "
                "do atendimento e comprovar que a solicitação foi registrada."
            )
        },

        "problema nao resolvido": {
            "palavras": [
                "problema nao resolvido",
                "problema não resolvido",
                "nao resolveram",
                "não resolveram",
                "reclamacao nao resolvida",
                "reclamação não resolvida"
            ],

            "resposta": (
                "⚠️ **Problema não resolvido**\n\n"
                "Se você já entrou em contato com a instituição "
                "e o problema não foi solucionado, reúna os "
                "protocolos e registros dos atendimentos anteriores.\n\n"

                "📌 Informe:\n"
                "• Número do protocolo;\n"
                "• Data do atendimento;\n"
                "• Setor procurado;\n"
                "• O que foi solicitado;\n"
                "• Por que o problema continua.\n\n"

                "Nesse caso, procure o **Atendimento ao Aluno** "
                "ou o canal responsável por reclamações e ouvidoria."
            )
        }
    }
}


# ============================================================
# NOVO - CONHECIMENTO GERAL
# ============================================================

conhecimento_geral = {

    "calendario": {
        "palavras": [
            "calendario",
            "calendario academico",
            "calendario escolar",
            "datas importantes",
            "datas da faculdade"
        ],

        "resposta": (
            "📅 **Calendário Acadêmico**\n\n"
            "O calendário acadêmico reúne as principais datas "
            "e prazos da faculdade durante o período letivo.\n\n"

            "Ele pode apresentar:\n"
            "• Início e término das aulas;\n"
            "• Matrículas;\n"
            "• Provas e avaliações;\n"
            "• Férias e recessos;\n"
            "• Prazos acadêmicos;\n"
            "• Outros eventos importantes.\n\n"

            "🔎 **Como consultar:**\n"
            "Acesse o portal acadêmico e procure por "
            "**Calendário Acadêmico**, **Calendário Escolar** "
            "ou **Datas Importantes**.\n\n"

            "🏢 Se não encontrar ou tiver dúvida sobre um prazo, "
            "procure a **Secretaria Acadêmica**."
        )
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
# NOVA FUNÇÃO - ENCONTRAR CONHECIMENTO ESPECÍFICO
# ============================================================

def encontrar_conhecimento_especifico(texto, categoria=None):

    texto_normalizado = normalizar_texto(texto)

    melhor_resposta = None
    melhor_pontuacao = 0

    # --------------------------------------------------------
    # PRIMEIRO: PROCURAR DENTRO DA CATEGORIA ATUAL
    # --------------------------------------------------------

    if categoria is not None and categoria in conhecimento_especifico:

        for nome_assunto, dados_assunto in conhecimento_especifico[categoria].items():

            pontuacao = 0

            for palavra in dados_assunto["palavras"]:

                palavra_normalizada = normalizar_texto(palavra)

                if palavra_normalizada in texto_normalizado:

                    quantidade_palavras = len(
                        palavra_normalizada.split()
                    )

                    if quantidade_palavras >= 2:
                        pontuacao += 5
                    else:
                        pontuacao += 2

            if pontuacao > melhor_pontuacao:

                melhor_pontuacao = pontuacao
                melhor_resposta = dados_assunto["resposta"]


    # --------------------------------------------------------
    # SEGUNDO: PROCURAR NA BASE GERAL
    # --------------------------------------------------------

    if melhor_resposta is None:

        for nome_assunto, dados_assunto in conhecimento_geral.items():

            pontuacao = 0

            for palavra in dados_assunto["palavras"]:

                palavra_normalizada = normalizar_texto(palavra)

                if palavra_normalizada in texto_normalizado:

                    quantidade_palavras = len(
                        palavra_normalizada.split()
                    )

                    if quantidade_palavras >= 2:
                        pontuacao += 5
                    else:
                        pontuacao += 2

            if pontuacao > melhor_pontuacao:

                melhor_pontuacao = pontuacao
                melhor_resposta = dados_assunto["resposta"]


    if melhor_pontuacao >= 2:

        return melhor_resposta

    return None


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
    # 5 - NOVO - PROCURAR RESPOSTA ESPECÍFICA
    # ========================================================

    resposta_especifica = encontrar_conhecimento_especifico(
        mensagem,
        categoria_por_palavra
    )


    if resposta_especifica is not None:

        # ----------------------------------------------------
        # IDENTIFICAR CATEGORIA QUANDO POSSÍVEL
        # ----------------------------------------------------

        if categoria_por_palavra is not None:

            categoria = categoria_por_palavra

            st.session_state.ultima_categoria = categoria

        elif st.session_state.ultima_categoria is not None:

            categoria = st.session_state.ultima_categoria

        else:

            categoria = None


        with st.chat_message("assistant"):

            st.write(resposta_especifica)

            if categoria is not None:

                st.divider()

                st.write(
                    f"🏷️ **Categoria identificada:** {categoria}"
                )

                st.write(
                    f"🏢 **Setor responsável:** {setores[categoria]}"
                )

            st.success(
                "✅ Espero que essa explicação tenha ajudado. "
                "Você pode continuar perguntando normalmente."
            )


        st.session_state.mensagens_chat.append(
            {
                "role": "assistant",
                "content": resposta_especifica
            }
        )


        st.stop()


    # ========================================================
    # 6 - CONTINUAÇÃO DA CONVERSA
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
    # 7 - CLASSIFICAÇÃO DO MODELO
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
    # 8 - ESCOLHER MELHOR CLASSIFICAÇÃO
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
    # 9 - SE NÃO HOUVER SEGURANÇA
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
    # 10 - SALVAR MEMÓRIA
    # ========================================================

    st.session_state.ultima_categoria = categoria


    # ========================================================
    # 11 - SETOR E SOLUÇÃO
    # ========================================================

    setor = setores[categoria]

    dados_solucao = solucoes[categoria]


    # ========================================================
    # 12 - DETECTAR NECESSIDADE DE HUMANO
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
    # 13 - RESPOSTA DO ASSISTENTE
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
