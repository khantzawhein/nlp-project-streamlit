import streamlit as st


def main():
    pg = st.navigation([st.Page("custom-pages/0-Home.py", title="Home", icon="🏠"),
                        st.Page("custom-pages/2-Jobs.py", title="Background Jobs", url_path="jobs", icon="⚙️"),
                        st.Page("custom-pages/1-Reports.py", title="Reports", url_path="reports", icon="📊")],)

    pg.run()


if __name__ == '__main__':
    main()
