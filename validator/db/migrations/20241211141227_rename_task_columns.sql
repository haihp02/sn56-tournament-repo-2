-- migrate:up
ALTER TABLE public.tasks RENAME COLUMN system TO field_system;
ALTER TABLE public.tasks RENAME COLUMN instruction TO field_instruction;
ALTER TABLE public.tasks RENAME COLUMN input TO field_input;
ALTER TABLE public.tasks RENAME COLUMN output TO field_output;
ALTER TABLE public.tasks RENAME COLUMN delay_times TO times_delayed;

ALTER TABLE public.tasks RENAME COLUMN end_timestamp TO termination_at;
ALTER TABLE public.tasks RENAME COLUMN created_timestamp TO created_at;
ALTER TABLE public.tasks RENAME COLUMN delay_timestamp TO next_delay_at;
ALTER TABLE public.tasks RENAME COLUMN updated_timestamp TO updated_at;
ALTER TABLE public.tasks RENAME COLUMN started_timestamp TO started_at;
ALTER TABLE public.tasks RENAME COLUMN completed_timestamp TO completed_at;

ALTER TABLE public.tasks
    ADD COLUMN system_format text DEFAULT NULL,
    ADD COLUMN trained_model_repository text DEFAULT NULL;


UPDATE public.tasks
SET user_id = '00000000-0000-0000-0000-000000000000'
WHERE user_id IS NULL;


-- change user_id to account_id and make it a non null uuid
ALTER TABLE public.tasks ALTER COLUMN user_id SET NOT NULL;

ALTER TABLE public.tasks
ALTER COLUMN user_id TYPE UUID
USING user_id::uuid;

ALTER TABLE public.tasks RENAME COLUMN user_id TO account_id;
CREATE INDEX idx_tasks_account_id ON public.tasks (account_id);

-- migrate:down
ALTER TABLE public.tasks RENAME COLUMN field_system TO system;
ALTER TABLE public.tasks RENAME COLUMN field_instruction TO instruction;
ALTER TABLE public.tasks RENAME COLUMN field_input TO input;
ALTER TABLE public.tasks RENAME COLUMN field_output TO output;
ALTER TABLE public.tasks RENAME COLUMN times_delayed TO delay_times;

ALTER TABLE public.tasks RENAME COLUMN termination_at TO end_timestamp;
ALTER TABLE public.tasks RENAME COLUMN created_at TO created_timestamp;
ALTER TABLE public.tasks RENAME COLUMN next_delay_at TO delay_timestamp;
ALTER TABLE public.tasks RENAME COLUMN updated_at TO updated_timestamp;
ALTER TABLE public.tasks RENAME COLUMN started_at TO started_timestamp;
ALTER TABLE public.tasks RENAME COLUMN completed_at TO completed_timestamp;

ALTER TABLE public.tasks
    DROP COLUMN system_format,
    DROP COLUMN trained_model_repository;


DROP INDEX idx_tasks_account_id;
ALTER TABLE public.tasks RENAME COLUMN account_id TO user_id;

ALTER TABLE public.tasks
ALTER COLUMN user_id TYPE TEXT
USING user_id::text;

ALTER TABLE public.tasks ALTER COLUMN user_id DROP NOT NULL;

UPDATE public.tasks
SET user_id = NULL
WHERE user_id = '00000000-0000-0000-0000-000000000000';
