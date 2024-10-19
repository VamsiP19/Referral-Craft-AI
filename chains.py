import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Chain:
    def __init__(self):
        # Set the API key explicitly, or you can use the os.getenv method if you prefer
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key="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",  # Explicit API key
            model_name="llama-3.1-70b-versatile"
        )

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills`, and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Mohan, a BTECH Computer Science student. You are looking for a job.
            Your task is to write a professional yet friendly message asking an employee of the company on the LinkedIn platform, who works in the same department, for a referral to a relevant hiring manager for a job you are interested in.
            Mention how excited you are about the company and briefly highlight your skills and experience that align with the job. Also, provide project links that are relevant for the job. Keep the tone appreciative and respectful, and avoid being too formal. 
            Your job is to write a referral message to the employee who is already working in the company regarding the job mentioned above, describing your capabilities 
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase the portfolio: {link_list}
            Remember you are Mohan, a computer science student.
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):

            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content


if __name__ == "__main__":
    print("GROQ API Key:", os.getenv("GROQ_API_KEY"))  # To verify if the key is loaded
