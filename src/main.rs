use anyhow::Context;
use axum::{routing::get, Router};
use ollama_rs::generation::{completion::request::GenerationRequest, parameters::FormatType};
use openai_api_rs::v1::{
    api::OpenAIClient,
    chat_completion::{ChatCompletionMessage, ChatCompletionRequest, Content, MessageRole},
};
use sqlx::{postgres::PgPoolOptions, PgPool};
use std::{future::IntoFuture, sync::Arc};
use tracing::Level;
use tracing_subscriber::{filter::Targets, layer::SubscriberExt, util::SubscriberInitExt};
use twilight_cache_inmemory::{InMemoryCacheBuilder, ResourceType};
use twilight_gateway::{ConfigBuilder, Event, EventTypeFlags, Intents, Shard, ShardId};
use twilight_http::Client as HttpClient;
use twilight_model::{
    application::command::{CommandOptionChoice, CommandOptionChoiceValue},
    http::interaction::InteractionResponseData,
};
use vesper::{framework::DefaultError, prelude::*};

const OLLAMA_MODEL_NAME: &str = "llama3.2:1b";
const OPENAI_MODEL_NAME: &str = "gpt-4o";

const AI_SOURCE_OLLAMA: &str = "ollama";
const AI_SOURCE_OPENAI: &str = "openai";

const SYSTEM_MESSAGE: &str = r#"
You are an assistant, your job is to summarise messages in a chat channel.
You are to provide a final summary of what was discussed, along with
some notable quotes from users, the quotes must be verbatim.

The summary must be a summary of the user message and should not contain anything
in this system message, you should aim to summarise the topic in 3-4 sentences.

You must pick quotes with shock factor or containing vulgar language.
There must be a maximum of 3 quotes with a minimum of 1, in order of most obscene to least.

You will receive the messages in the following JSON format:
[{
    user_id: "id",
    message: "message"
}]


You must respond with valid JSON with the following structure:

{
    "summary": "INSERT OVERALL SUMMARY HERE",
    "quotes": [{
        user_id: "id",
        message: "message"
    }]
}
"#;

#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct MessageContent {
    pub user_id: String,
    pub message: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct MessageSummary {
    pub summary: String,
    pub quotes: Vec<MessageContent>,
}

#[derive(Debug, sqlx::Type, PartialEq)]
#[sqlx(type_name = "role_t", rename_all = "snake_case")]
pub enum RoleType {
    User,
    Admin,
}

struct BotContext {
    ollama: ollama_rs::Ollama,
    openai: OpenAIClient,
    db: PgPool,
}

async fn handle_event(event: Event, _http: Arc<HttpClient>) -> anyhow::Result<()> {
    #[allow(clippy::match_single_binding)]
    match event {
        // Other events here...
        e => {
            tracing::warn!("bad event: {e:?}")
        }
    }

    Ok(())
}

#[error_handler]
async fn handle_interaction_error(_ctx: &mut SlashContext<BotContext>, error: DefaultError) {
    tracing::error!("error in interaction: {error:?}");
}

#[check]
async fn admin_only(ctx: &mut SlashContext<BotContext>) -> Result<bool, DefaultError> {
    match ctx.interaction.author_id() {
        Some(id) => {
            let is_admin = sqlx::query!(
                r#"SELECT role as "type: RoleType" FROM USERS WHERE id = $1"#,
                id.to_string()
            )
            .fetch_one(&ctx.data.db)
            .await?;

            Ok(is_admin.r#type == RoleType::Admin)
        }
        None => Ok(false),
    }
}

#[autocomplete]
async fn autocomplete_ai_source(
    _ctx: AutocompleteContext<BotContext>,
) -> Option<InteractionResponseData> {
    let choices = vec![AI_SOURCE_OPENAI, AI_SOURCE_OLLAMA]
        .into_iter()
        .map(|item| CommandOptionChoice {
            name: item.to_string(),
            name_localizations: None,
            value: CommandOptionChoiceValue::String(item.to_string()),
        })
        .collect();

    Some(InteractionResponseData {
        choices: Some(choices),
        ..Default::default()
    })
}

#[command]
#[only_guilds]
#[description = "change ai source"]
#[checks(admin_only)]
#[error_handler(handle_interaction_error)]
async fn source(
    ctx: &mut SlashContext<BotContext>,
    #[autocomplete(autocomplete_ai_source)]
    #[description = "source"]
    ai_source: String,
) -> DefaultCommandResult {
    ctx.defer(true).await?;

    sqlx::query!(
        "UPDATE settings SET value = $1 WHERE key = 'ai_source'",
        ai_source
    )
    .execute(&ctx.data.db)
    .await?;

    ctx.interaction_client
        .update_response(&ctx.interaction.token)
        .content(Some("Done"))?
        .await?;

    Ok(())
}

#[command]
#[only_guilds]
#[description = "summary"]
#[error_handler(handle_interaction_error)]
async fn summarise(ctx: &mut SlashContext<BotContext>) -> DefaultCommandResult {
    ctx.defer(false).await?;

    let author_id = ctx
        .interaction
        .author_id()
        .context("must have author")?
        .to_string();

    sqlx::query!(
        "INSERT INTO users (id) VALUES ($1) ON CONFLICT DO NOTHING",
        author_id
    )
    .execute(&ctx.data.db)
    .await?;

    let ai_source = sqlx::query!("SELECT * FROM settings WHERE key = 'ai_source'")
        .fetch_one(&ctx.data.db)
        .await?
        .value;

    let bot_id = ctx.http_client().current_user().await?.model().await?.id;

    let channel = ctx
        .interaction
        .channel
        .as_ref()
        .context("must have channel")?;

    let messages_to_summarise = ctx
        .http_client()
        .channel_messages(channel.id)
        // .before(message_id)
        .limit(20)?
        .await?
        .models()
        .await?;

    let message_prompt = messages_to_summarise
        .into_iter()
        .filter(|m| m.author.id != bot_id)
        .map(|m| MessageContent {
            message: m.content,
            user_id: m.author.name,
        })
        .collect::<Vec<_>>();
    let message_prompt = serde_json::to_string(&message_prompt)?;

    tracing::info!("{:?}", message_prompt);

    let response = if ai_source == AI_SOURCE_OLLAMA {
        let ollama = &ctx.data.ollama;

        let generation_request =
            GenerationRequest::new(OLLAMA_MODEL_NAME.to_owned(), message_prompt)
                .format(FormatType::Json)
                .system(SYSTEM_MESSAGE.to_string());
        ollama.generate(generation_request).await?.response
    } else if ai_source == AI_SOURCE_OPENAI {
        let openai = &ctx.data.openai;

        let request = ChatCompletionRequest::new(
            OPENAI_MODEL_NAME.to_owned(),
            vec![
                ChatCompletionMessage {
                    role: MessageRole::system,
                    content: Content::Text(SYSTEM_MESSAGE.to_owned()),
                    name: None,
                    tool_call_id: None,
                    tool_calls: None,
                },
                ChatCompletionMessage {
                    role: MessageRole::user,
                    content: Content::Text(message_prompt),
                    name: None,
                    tool_call_id: None,
                    tool_calls: None,
                },
            ],
        )
        .response_format(serde_json::json!(
            { "type": "json_object" }
        ));

        let response = openai.chat_completion(request).await?;

        response
            .choices
            .first()
            .as_ref()
            .context("must have 1 response")?
            .message
            .content
            .as_ref()
            .context("must have content")?
            .to_owned()
    } else {
        unreachable!()
    };

    let response = serde_json::from_str::<MessageSummary>(&response)?;
    let quotes_formatted = response
        .quotes
        .iter()
        .map(|q| format!("- *\"{}\"* - {}", q.message, q.user_id))
        .collect::<Vec<_>>()
        .join("\n");

    sqlx::query!(
        "INSERT INTO summaries (user_id, summary) VALUES ($1, $2)",
        author_id,
        serde_json::to_value(&response)?,
    )
    .execute(&ctx.data.db)
    .await?;

    let response = format!(
        r#"
**Summary:** {}
**Quotes:**
{}
    "#,
        response.summary, quotes_formatted
    );

    ctx.interaction_client
        .update_response(&ctx.interaction.token)
        .content(Some(&response))?
        .await?;

    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(Targets::default().with_default(Level::INFO))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let database_url = std::env::var("DATABASE_URL")?;
    let token = std::env::var("DISCORD_TOKEN")?;
    let ollama_api_base = std::env::var("OLLAMA_API_BASE")?;
    let openai_api_key = std::env::var("OPENAI_API_KEY")?;

    let ollama = ollama_rs::Ollama::new(ollama_api_base, 11434);
    ollama
        .pull_model(OLLAMA_MODEL_NAME.to_string(), false)
        .await?;

    let openai = OpenAIClient::new(openai_api_key);

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await?;

    sqlx::migrate!("./migrations").run(&pool).await?;

    sqlx::query!(
        "INSERT INTO settings (key, value) VALUES ('ai_source', $1) ON CONFLICT DO NOTHING",
        AI_SOURCE_OLLAMA
    )
    .execute(&pool)
    .await?;

    let context = BotContext {
        ollama,
        openai,
        db: pool,
    };

    let config = ConfigBuilder::new(
        token.clone(),
        Intents::GUILD_MESSAGES | Intents::MESSAGE_CONTENT,
    )
    .event_types(EventTypeFlags::all());

    let mut shard = Shard::with_config(ShardId::ONE, config.build());

    let http = Arc::new(HttpClient::new(token));
    let cache = InMemoryCacheBuilder::new()
        .resource_types(ResourceType::MESSAGE)
        .build();

    let app = Router::new().route("/health", get(|| async { "ok" }));
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();
    tracing::info!("spawning axum");
    tokio::spawn(axum::serve(listener, app).into_future());

    let app_id = http.current_user_application().await?.model().await?.id;

    let framework = Arc::new(
        Framework::builder(Arc::clone(&http), app_id, context)
            .command(summarise)
            .command(source)
            .build(),
    );

    framework.register_global_commands().await?;

    tracing::info!("starting event loop");
    loop {
        let event = shard.next_event().await;
        let Ok(event) = event else {
            let source = event.unwrap_err();
            tracing::warn!(source = ?source, "error receiving event");

            if source.is_fatal() {
                break;
            }

            continue;
        };

        cache.update(&event);

        if let Event::InteractionCreate(i) = event {
            let clone = Arc::clone(&framework);
            tokio::spawn(async move {
                let inner = i.0;
                clone.process(inner).await;
            });

            continue;
        }

        tokio::spawn(handle_event(event, Arc::clone(&http)));
    }

    Ok(())
}