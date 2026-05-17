
import streamlit as st
import pandas as pd


def render_qa_page(qa_system, formula_engine):
    """Отображение страницы вопрос-ответ"""
    st.subheader("💬 Задайте вопрос по документации")

    # Вкладки
    qa_tab1, qa_tab2, qa_tab3 = st.tabs([
        "📝 Общие вопросы",
        "📐 Расчеты и формулы",
        "📊 Калькулятор"
    ])

    with qa_tab1:
        st.markdown("""
        **Примеры вопросов:**
        - Какие материалы можно использовать для тепловой изоляции?
        - Какая толщина изоляции для трубопроводов?
        - Какие требования к пароизоляционному слою?
        - Какие нормативы регулируют тепловую изоляцию?
        - Что такое коэффициент уплотнения?
        """)

        question = st.text_input(
            "Введите ваш вопрос:",
            placeholder="Например: Какие материалы для изоляции?"
        )

        if st.button("🔍 Найти ответ", key="search_btn"):
            with st.spinner("🔍 Ищу ответ..."):
                result = qa_system.answer(question)

                st.markdown("---")
                st.subheader("📝 Ответ")
                st.markdown(result['answer'])

                if result.get('sources'):
                    with st.expander("📚 Источники"):
                        for i, src in enumerate(result['sources'][:3], 1):
                            st.write(f"**{i}. {src['doc_name']}** (релевантность: {src['score']:.2f})")
                            st.write(src['text'][:300])

    with qa_tab2:
        st.markdown("""
        **Вопросы по расчетам и формулам:**
        - Как рассчитать толщину тепловой изоляции?
        - Как рассчитать тепловые потери?
        - Какая температура будет на поверхности изоляции?
        - Какие нормы плотности теплового потока?
        """)

        calc_question = st.text_input(
            "Введите расчетный вопрос:",
            placeholder="Например: Рассчитай толщину изоляции при температуре 150°C"
        )

        if st.button("🧮 Рассчитать", key="calc_btn"):
            with st.spinner("Выполняю расчет..."):
                result = formula_engine.answer_calculation(calc_question)

                st.markdown("---")
                st.subheader("📐 Результат расчета")
                st.markdown(result['answer'])

    with qa_tab3:
        st.subheader("🧮 Калькулятор толщины изоляции")

        col1, col2 = st.columns(2)

        with col1:
            t_in = st.number_input("Температура теплоносителя (°C)", value=150, min_value=-100, max_value=600)
            t_out = st.number_input("Температура окружающей среды (°C)", value=20, min_value=-50, max_value=50)
            lam = st.number_input("Теплопроводность изоляции (Вт/м·°C)", value=0.045, min_value=0.01, max_value=0.2, step=0.005)

        with col2:
            q_norm = st.number_input("Нормированная плотность потока (Вт/м²)", value=40, min_value=10, max_value=200)
            R_outer = st.number_input("Сопротивление теплоотдаче (м²·°C/Вт)", value=0.1, min_value=0.05, max_value=0.3, step=0.01)

        if st.button("📐 Рассчитать", key="calc_thickness"):
            delta = lam * ((t_in - t_out) / q_norm - R_outer)
            delta_mm = delta * 1000

            st.markdown("---")
            st.subheader("📊 Результат расчета")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Расчетная толщина", f"{delta:.3f} м")
            with col2:
                st.metric("Расчетная толщина", f"{delta_mm:.0f} мм")
            with col3:
                st.metric("Рекомендуемая", f"{round(delta_mm/10)*10} мм")

            st.markdown("**Формула расчета (СП 61.13330.2012, формула В.18):**")
            st.code("δ = λ × ((tв - tн) / q - Rн)", language="text")

            st.markdown("**Подстановка:**")
            st.text(f"""
δ = {lam} × (({t_in} - {t_out}) / {q_norm} - {R_outer})
δ = {lam} × ({t_in - t_out} / {q_norm} - {R_outer})
δ = {lam} × ({(t_in - t_out) / q_norm:.3f} - {R_outer})
δ = {lam} × {((t_in - t_out) / q_norm - R_outer):.4f}
δ = {delta:.4f} м = {delta_mm:.0f} мм
""")

            # Нормативная проверка
            if delta_mm < 20:
                st.warning("⚠️ **Внимание:** Расчетная толщина менее 20 мм. Согласно п. 6.13, минимальная толщина изоляции должна быть не менее 20 мм. Рекомендуется принять 20 мм.")
            elif delta_mm > 320:
                st.warning("⚠️ **Внимание:** Расчетная толщина превышает 320 мм. Согласно приложению Г, это максимальная толщина для трубопроводов. Рассмотрите применение более эффективной изоляции.")
            elif delta_mm > 200:
                st.info("ℹ️ Расчетная толщина превышает 200 мм. Рекомендуется рассмотреть многослойную конструкцию изоляции (п. 6.16).")
            else:
                st.success("✅ Расчетная толщина соответствует нормативным требованиям.")

            with st.expander("📖 Подробнее о расчете"):
                st.markdown("""
                **СП 61.13330.2012, п. 6.2.1 - Расчет толщины изоляции по нормированной плотности теплового потока**

                Формула В.18 для плоских и цилиндрических поверхностей с диаметром 1,4 м и более:
δ = λ × ((tв - tн) / q - Rн)

text

**Где:**
- δ — толщина изоляции (м)
- λ — теплопроводность материала в конструкции (Вт/м·°C)
- tв — температура теплоносителя (°C)
- tн — температура окружающей среды (°C)
- q — нормированная плотность теплового потока (Вт/м²)
- Rн — сопротивление теплоотдаче на наружной поверхности (м²·°C/Вт)

**Нормативные значения:**
- Минимальная толщина: 20 мм (п. 6.13)
- Максимальная толщина: 320 мм (приложение Г)
- Коэффициент теплоотдачи αн: 11 Вт/м²·°C (таблица В.2)

**Для трубопроводов** (диаметром менее 1,4 м) используется формула В.20:
δ = (d/2) × (exp(B) - 1)

text
где B = 2π × λ × (tв - tн) / q
""")