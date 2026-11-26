p = r"C:\ARGUS-X2\frontend\app.py"
s = open(p, encoding="utf-8").read()

old1 = '''        st.dataframe(pd.DataFrame(ss).drop(columns=["output"], errors="ignore").head(8), hide_index=True,
                     use_container_width=True) if ss else st.info("No sessions yet. Process a video first.")'''
new1 = '''        if ss:
            st.dataframe(pd.DataFrame(ss).drop(columns=["output"], errors="ignore").head(8), hide_index=True,
                         use_container_width=True)
        else:
            st.info("No sessions yet. Process a video first.")'''

old2 = '''        st.dataframe(pd.DataFrame(events).tail(10).iloc[::-1], hide_index=True, use_container_width=True) \\
            if events else st.info("No identity events yet.")'''
new2 = '''        if events:
            st.dataframe(pd.DataFrame(events).tail(10).iloc[::-1], hide_index=True, use_container_width=True)
        else:
            st.info("No identity events yet.")'''

old3 = '''        st.dataframe(pd.DataFrame(ev), hide_index=True, use_container_width=True) if ev else st.caption("No events.")'''
new3 = '''        if ev:
            st.dataframe(pd.DataFrame(ev), hide_index=True, use_container_width=True)
        else:
            st.caption("No events in this session.")'''

for i, (o, n) in enumerate([(old1, new1), (old2, new2), (old3, new3)], 1):
    if o in s:
        s = s.replace(o, n)
        print("patched block", i)
    else:
        print("NOT FOUND: block", i)

open(p, "w", encoding="utf-8").write(s)
